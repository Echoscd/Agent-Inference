# algorithm/

The experiment harness: the agent, the concurrent driver, the metric layer and
the figures. The scheduling policies themselves live in `../ThunderAgent/`.

Nothing here hardcodes a machine path — `paths.py` resolves everything from the
repo root, overridable with `AGENT_EXP_ROOT`, `AGENT_EXP_RESULT`,
`AGENT_EXP_WORK_ROOT`, `AGENT_EXP_DATA`, `AGENT_EXP_CONDA`.

Documentation for this code lives in `../docs/`:
[architecture](../docs/architecture.md) (pipeline, entry points, A/B design),
[metrics](../docs/metrics.md) (every metric definition),
[reproducing](../docs/reproducing.md) (how to run one).

## Files

- `paths.py` — repo-relative paths, conda auto-detection
- `run_metrics.py` / `test_run_metrics.py` — metric definitions and their tests
- `swebench_agent.py` — shared dataclasses (`TurnTiming`, `AgentResult`), config,
  and the bash-style ReAct agent (`--agent bash`, not used by the reported runs)
- `swebench_edit_agent.py` — the edit agent used in every reported experiment
- `swebench_local_harness.py` — Docker-free conda harness: build, apply patch,
  evaluate, reset. `--patch gold` validates an instance.
- `metrics.py` — vLLM `/metrics`: in-process `MetricsMonitor` plus a standalone
  sampler (`python3 metrics.py <out.csv> [url] [interval]`)
- `ab_tape.py` — the tape: env-gated recording (`AB_RECORD_TAPE`) and
  known-decode lookup (`AB_KNOWN_DECODE_TAPE`)
- `legacy/` — superseded HumanEval pipeline, kept for reference only

## History

The scheduling code was refactored out of the router into policy classes
(`ThunderAgent/ThunderAgent/scheduling/`), and the metric code was unified into
`run_metrics.py`. Earlier revisions of this file described `_program_density` and
"density branches in `_greedy_resume`" inside `router.py`; those no longer exist.
