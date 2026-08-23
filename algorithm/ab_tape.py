"""Record / replay tape for the A/B serving experiment (run_AB_experiment.sh).

Both passes run the REAL edit-agent at temperature=0, so the trajectory is
deterministic and reproduces turn-for-turn. Two env vars control the tape:

  AB_RECORD_TAPE=<path>        pass writes one JSON record per LLM call:
                               {iid, turn, t_start_s, prompt_tokens, gen_tokens,
                                wait_s, decode_s, tool_wait_s, messages, completion}
                               i.e. the full prefill (messages) and decode (completion)
                               content is saved for every call.

  AB_KNOWN_DECODE_TAPE=<path>  pass reads a pass-1 tape; known_decode(iid, turn) returns
                               that call's decode length so the agent can send
                               X-Decode-Len to ThunderAgent (the density policy's
                               "known decode"). Falls back to None if absent.

Pass 1: AB_RECORD_TAPE=tape_A.jsonl  (size policy, baseline).
Pass 2: AB_RECORD_TAPE=tape_B.jsonl AB_KNOWN_DECODE_TAPE=tape_A.jsonl  (density policy).
Comparing tape_A vs tape_B verifies the decode reproduced exactly.
"""
import os
import json
import threading
import time

_RECORD_PATH = os.environ.get("AB_RECORD_TAPE")
_KNOWN_PATH = os.environ.get("AB_KNOWN_DECODE_TAPE")

# Run clock: t=0 is the moment this module is imported, i.e. the start of the
# agent run. Every recorded call carries t_start_s = seconds since then, so the
# analysis no longer has to reconstruct absolute time by cumsumming per-program
# durations (which silently drops client-side build/eval time between turns).
T0 = time.perf_counter()


def now() -> float:
    """Seconds since the run started."""
    return time.perf_counter() - T0


_write_lock = threading.Lock()
_load_lock = threading.Lock()
_known = None  # lazy { (iid, turn): gen_tokens }


def recording() -> bool:
    return bool(_RECORD_PATH)


def record_call(iid: str, turn: int, messages, completion: str,
                prompt_tokens: int, gen_tokens: int,
                wait_s: float = 0.0, decode_s: float = 0.0, tool_wait_s: float = 0.0,
                t_start_s: float = None) -> None:
    """Append one call's full prefill+decode content + timing to the record tape (no-op if unset).

    Recorded for EVERY call as it happens, so the tape captures all programs incl.
    ones that never finish (timed out) — unlike results.jsonl which only has finishers.
    Timing lets per-call stats / Gantt cover the full population.
    """
    if not _RECORD_PATH:
        return
    rec = {
        "iid": iid,
        "turn": turn,
        "prompt_tokens": int(prompt_tokens),
        "gen_tokens": int(gen_tokens),
        "wait_s": round(float(wait_s), 3),       # TTFT = proxy pause + queue + prefill
        "decode_s": round(float(decode_s), 3),   # generation (running) time
        "tool_wait_s": round(float(tool_wait_s), 3),
        # absolute position on the run timeline (s since run start); consumed by
        # run_metrics.RunArtifacts, which falls back to reconstruction if absent
        "t_start_s": round(float(t_start_s if t_start_s is not None else now()), 3),
        "messages": messages,
        "completion": completion,
    }
    line = json.dumps(rec, ensure_ascii=False)
    with _write_lock:
        with open(_RECORD_PATH, "a") as f:
            f.write(line + "\n")


def _load_known() -> None:
    global _known
    _known = {}
    if not _KNOWN_PATH or not os.path.exists(_KNOWN_PATH):
        return
    for line in open(_KNOWN_PATH):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        _known[(r["iid"], int(r["turn"]))] = int(r.get("gen_tokens", 0))


def known_decode(iid: str, turn: int):
    """Return the decode length recorded for (iid, turn) in the known-decode tape, else None."""
    global _known
    if not _KNOWN_PATH:
        return None
    if _known is None:
        with _load_lock:
            if _known is None:
                _load_known()
    return _known.get((iid, int(turn)))
