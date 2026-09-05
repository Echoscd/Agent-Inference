# Documentation

One topic, one file. If something is documented in two places it will drift, so
each page below owns its subject and everything else links to it.

| page | owns |
|---|---|
| [reproducing.md](reproducing.md) | setup, what gets downloaded, how to run an A/B, what a fresh clone can recompute |
| [architecture.md](architecture.md) | the pipeline, the A/B design, how policies plug in |
| [metrics.md](metrics.md) | every metric definition: latency notions, windows, the warmup rule, percentiles |
| [experiments.md](experiments.md) | what each `result/` folder is, including which ones are invalid |
| [limitations.md](limitations.md) | what these results do not support, and why |
| [decision_traces.md](decision_traces.md) | the per-tick scheduler traces: record format and how to read them |

Not here, on purpose:

- `../result/33_warmup_steady_metrics/README.md` and other per-experiment notes
  stay next to their data.
- `../sim/` keeps its own docs; the simulator is a separate artifact with its own
  settings and calibration write-ups.
- `../ThunderAgent/` is vendored upstream documentation, unmodified.

The weekly log of what was actually done lives in [../worklog/](../worklog/).
