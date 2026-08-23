"""Tests for the unified metric definitions (run_metrics.py).

These pin the semantics that used to be re-implemented per script: the
percentile formula, overlap-weighted throughput, which calls a window's latency
covers, the warmup rule, and recorded-vs-reconstructed timelines.
"""
import json
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_metrics as rm


# ── percentiles ───────────────────────────────────────────────────────────────
def test_pctl_interpolates_like_numpy_linear():
    xs = list(range(1, 101))
    assert rm.pctl(xs, 50) == pytest.approx(50.5)
    assert rm.pctl(xs, 95) == pytest.approx(95.05)
    assert rm.pctl(xs, 99) == pytest.approx(99.01)


def test_pctl_handles_empty_single_and_unsorted():
    assert rm.pctl([], 95) == 0.0
    assert rm.pctl([7], 95) == 7
    assert rm.pctl([5, 1, 3], 50) == rm.pctl([1, 3, 5], 50) == 3


def test_dist_block_has_p95_and_drops_none():
    d = rm.dist([None, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    assert set(d) == {"min", "median", "p90", "p95", "p99", "mean", "max"}
    assert (d["min"], d["max"], d["median"]) == (1, 10, 5.5)
    assert d["p90"] <= d["p95"] <= d["p99"]


# ── window attribution ────────────────────────────────────────────────────────
def _call(start, wait, decode, gen, iid="p", turn=1, tool=0.0):
    return rm.Call(iid=iid, turn=turn, start=start, decode_start=start + wait,
                   end=start + wait + decode, wait_s=wait, decode_s=decode,
                   tool_wait_s=tool, gen_tokens=gen, prompt_tokens=0)


def test_call_latency_excludes_tool_time():
    assert _call(0, 2, 3, 10, tool=99).latency_s == 5


def test_throughput_is_overlap_weighted_at_the_edge():
    # decode spans [10, 20) with 100 tokens; window covers the second half only
    c = _call(start=10, wait=0, decode=10, gen=100)
    assert rm.Window("w", 15, 100).gen_tokens([c]) == pytest.approx(50)
    assert rm.Window("w", 0, 100).gen_tokens([c]) == pytest.approx(100)
    assert rm.Window("w", 30, 100).gen_tokens([c]) == pytest.approx(0)


def test_zero_length_decode_is_credited_once():
    c = _call(start=5, wait=0, decode=0, gen=7)
    assert rm.Window("w", 0, 10).gen_tokens([c]) == 7
    assert rm.Window("w", 6, 10).gen_tokens([c]) == 0


def test_latency_covers_calls_that_START_inside():
    early = _call(start=0, wait=1, decode=100, gen=9000)   # still running in-window
    inside = _call(start=50, wait=1, decode=1, gen=10)
    m = rm.Window("w", 10, 100).metrics([early, inside])
    assert m["calls_started"] == 1                         # `early` started before t0
    assert m["latency_s"]["p50"] == pytest.approx(2)       # only `inside` counts
    # `early` contributes no latency sample but its in-window decode still counts:
    # 90s of its 100s decode overlap -> 0.9 * 9000 tokens, plus `inside`'s 10.
    assert m["gen_tokens_in_window"] == pytest.approx(8110)


# ── warmup rule ───────────────────────────────────────────────────────────────
def _kv(pairs):
    return rm.KvSeries([p[0] for p in pairs], [p[1] for p in pairs],
                       [0] * len(pairs), [0] * len(pairs), [0] * len(pairs), [0] * len(pairs))


def test_warm_edges_are_first_and_last_crossing_of_90pct_of_peak():
    kv = _kv([(0, 0), (10, 50), (20, 100), (30, 95), (40, 40), (50, 10)])
    assert kv.warm_edges() == (20, 30)     # 90% of peak 100 -> >= 90
    assert kv.peak_kv == 100


def test_warm_edges_scale_with_the_runs_own_peak():
    """Relative threshold: an arm that never fills KV must not be judged against
    another arm's peak, or the warmup length itself becomes the compared quantity."""
    assert _kv([(0, 0), (10, 40), (20, 50), (30, 20)]).warm_edges() == (20, 20)


def test_warm_edges_on_empty_series():
    assert _kv([]).warm_edges() == (0.0, 0.0)


# ── artifacts / timeline ──────────────────────────────────────────────────────
def _write_run(tmp_path, recs, kv_rows=((0, 0), (60, 100))):
    (tmp_path / "tape_A.jsonl").write_text("\n".join(json.dumps(r) for r in recs))
    lines = ["t_s,kv_perc,running,waiting,prefix_hit_cum,preemptions"]
    lines += [f"{t},{k},1,0,0,0" for t, k in kv_rows]
    (tmp_path / "kv_A.csv").write_text("\n".join(lines))
    return rm.RunArtifacts(str(tmp_path), "A")


BASE = {"wait_s": 1, "decode_s": 2, "tool_wait_s": 5, "gen_tokens": 10, "prompt_tokens": 100}


def test_recorded_timestamps_win_over_reconstruction(tmp_path):
    a = _write_run(tmp_path, [dict(BASE, iid="p1", turn=1, t_start_s=0.0),
                              dict(BASE, iid="p1", turn=2, t_start_s=50.0)])
    assert [c.start for c in a.calls] == [0.0, 50.0]
    assert a.timeline == "recorded"


def test_missing_timestamps_fall_back_to_cumsum(tmp_path):
    a = _write_run(tmp_path, [dict(BASE, iid="p1", turn=1),
                              dict(BASE, iid="p1", turn=2)])
    assert [c.start for c in a.calls] == [0.0, 8.0]   # 1 + 2 + 5
    assert a.timeline == "reconstructed"


def test_turns_are_ordered_and_programs_independent(tmp_path):
    a = _write_run(tmp_path, [dict(BASE, iid="p2", turn=2),
                              dict(BASE, iid="p1", turn=1),
                              dict(BASE, iid="p2", turn=1)])
    assert len(a.calls) == 3
    assert [c.start for c in a.calls if c.iid == "p2"] == [0.0, 8.0]
    assert [c.start for c in a.calls if c.iid == "p1"] == [0.0]


def test_corrupt_tape_lines_are_skipped(tmp_path):
    (tmp_path / "tape_A.jsonl").write_text(
        json.dumps(dict(BASE, iid="p1", turn=1)) + "\n{ truncated\n\n")
    (tmp_path / "kv_A.csv").write_text("t_s,kv_perc,running,waiting,prefix_hit_cum,preemptions\n0,0,0,0,0,0\n")
    assert len(rm.RunArtifacts(str(tmp_path), "A").calls) == 1


def test_three_standard_windows_and_row_shape(tmp_path):
    a = _write_run(tmp_path, [dict(BASE, iid="p1", turn=1), dict(BASE, iid="p1", turn=2)],
                   kv_rows=((0, 0), (5, 100), (20, 100), (30, 10)))
    names = [w.name for w in a.windows()]
    assert names == ["full", "warm", "steady"]
    full, warm, steady = a.windows()
    assert (full.t0, warm.t0, steady.t0) == (0.0, 5.0, 5.0)
    assert steady.t1 == 20.0 and warm.t1 == full.t1 == a.wall_s
    assert len(rm.RunMetrics(a).to_row()) == len(rm.RunMetrics.columns())
