#!/usr/bin/env python3
"""Figure for result/33_warmup_steady_metrics: full-run vs post-warmup vs steady."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import paths

def main():
    OUT = paths.result("33_warmup_steady_metrics")
    rows = json.load(open(f"{OUT}/steady_metrics.json"))
    # drop aborted arms (no policy recorded / crashed early)
    BAD = {("19_AB_fidelity_vs_size", "A"), ("19_AB_fidelity_vs_size", "B"),
           ("25_AB_size_vs_hazard", "B")}
    rows = [r for r in rows if (r["run"], r["arm"]) not in BAD]

    labels = [f"{r['run'].split('_')[0]}{r['arm']}\n{r['policy'] or '?'}" for r in rows]
    x = np.arange(len(rows))
    w = 0.27
    fig, ax = plt.subplots(3, 1, figsize=(15, 12), sharex=True)

    for i, (win, c) in enumerate((("full", "#9aa5b1"), ("warm", "#2b6cb0"), ("steady", "#dd6b20"))):
        ax[0].bar(x + (i - 1) * w, [r[win]["gen_tps"] for r in rows], w, label=win, color=c)
        ax[1].bar(x + (i - 1) * w, [r[win]["latency_s"]["p90"] for r in rows], w, label=win, color=c)
        ax[2].bar(x + (i - 1) * w, [r[win]["latency_s"]["p95"] for r in rows], w, label=win, color=c)

    ax[0].set_ylabel("generation throughput (tok/s)")
    ax[0].set_title("Post-warmup throughput and tail latency per LLM call (full run vs warmup dropped vs warmup+drain dropped)")
    ax[1].set_ylabel("p90 call latency (s)")
    ax[2].set_ylabel("p95 call latency (s)")
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(labels, fontsize=8)
    for a in ax:
        a.grid(axis="y", alpha=0.3)
        a.legend(fontsize=8)
        a.set_yscale("log") if a is not ax[0] else None
    fig.tight_layout()
    fig.savefig(f"{OUT}/warmup_vs_full.png", dpi=130)
    print("wrote", f"{OUT}/warmup_vs_full.png")

    # --- per-run A/B deltas on the post-warmup window -------------------------
    pairs = {}
    for r in rows:
        pairs.setdefault(r["run"], {})[r["arm"]] = r
    lines = []
    for run, ab in sorted(pairs.items()):
        if "A" not in ab or "B" not in ab:
            continue
        a, b = ab["A"], ab["B"]
        for win in ("warm", "steady"):
            ta, tb = a[win]["gen_tps"], b[win]["gen_tps"]
            pa, pb = a[win]["latency_s"]["p95"], b[win]["latency_s"]["p95"]
            lines.append(f"{run:<32} {win:<7} {a['policy']}->{b['policy']}: "
                         f"tps {ta:7.1f}->{tb:7.1f} ({100*(tb-ta)/ta:+6.1f}%)  "
                         f"p95 {pa:7.1f}->{pb:7.1f} ({100*(pb-pa)/pa:+6.1f}%)")
    open(f"{OUT}/ab_delta.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
