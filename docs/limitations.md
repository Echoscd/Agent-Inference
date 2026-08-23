# Known limitations

What these results do not support, and why.


- **Trajectory drift.** Both arms run at temperature 0 and should reproduce
  turn for turn, but vLLM batch variance makes them diverge (typically ~15% of
  (program, turn) pairs differ). The A/B is therefore not a perfectly paired
  comparison; `run_AB_experiment.sh` prints the drift at the end of every run.
- **Absolute call timestamps** were added to the tape on 2026-08-23. Runs before
  that reconstruct the timeline by cumsumming per-turn durations, which omits
  client-side build/eval time between turns; each row's `timeline` field says
  which applied.
- **Replicate count.** The size-vs-hazard_grade comparison currently has three
  replicates (26, 27, 28). Throughput and tail-latency deltas are not significant
  at that n; the resolve-rate difference is consistent across all three.
- Per-turn `max_tokens` is 8192 and context is capped at 40960, so long agents
  end with a 400 from the server rather than a graceful stop. Those programs are
  in the stats (status `NO_PATCH` with an error string), by design.
