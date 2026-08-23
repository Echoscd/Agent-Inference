# Simulator vs the real system: where they differ

**Status: skeleton, awaiting the reference material.** The sections below are the
questions this document needs to answer; the answers are not written yet because
the comparison should be made against the reference you are going to provide,
not against my reading of the simulator alone.

What is already known and can be filled in immediately is marked *(have)*; what
needs your input is marked *(need)*.

## 1. What each side is

- *(have)* Real system: 80 SWE-bench edit agents against one vLLM 0.12.0 backend,
  routed through the ThunderAgent proxy, which decides which program holds KV.
  Code in `algorithm/` + `ThunderAgent/`, results in `result/`.
- *(have)* Simulator: `sim/code/realistic_agentic_sim.py`, with its settings in
  `sim/settings/` and its own write-ups in `sim/SIMULATOR_SETTING.md` and
  `sim/SIMULATOR_CALIBRATION_EXPERIMENTS.md`.
- *(need)* Which simulator configuration is the one to compare against — the repo
  ships `real_proxy.json` and `synthetic_heldout.json`.

## 2. What the simulator models that the real system also has

*(need)* — the point-by-point mapping. Candidates to check: admission and
eviction decisions, KV footprint per program, tool-call gaps between turns,
program arrival, preemption.

## 3. What the simulator leaves out

*(need)* the authoritative list. From the real runs, these are the effects a
simulator has to either model or explicitly disclaim:

- prefix-cache hit rate, which collapses from ~93% to ~40% under eviction
  thrashing and is a first-order effect on prefill cost
- recompute-preemption: vLLM 0.12 V1 preempts by recompute, not swap
- batching effects: decode speed depends on how many sequences share a step
- the 8192-token per-turn cap and the 40960 context limit, which end long
  programs with a server-side 400 rather than a clean stop
- non-determinism: two arms at temperature 0 still drift apart

## 4. Where the numbers disagree

*(need)* the specific comparison you want made. The real-side numbers are
available for: throughput, per-call latency distribution, KV utilisation over
time, running/waiting concurrency, preemption counts, prefix-cache hit rate,
resolve rate. The simulator side has `sim/results/trace_real_proxy/metrics.csv`
and the decision-level traces next to it.

Related and already measured: a separate simulator in the Nips-AF work was found
to be arrival-starved at its default generation probability, which alone
accounted for a −36% gap against theory. Worth checking whether the equivalent
knob here is set into a comparable regime before attributing any gap to modelling.

## 5. What the simulator is still good for

*(need)* — a fair statement of what it buys, given the differences above. It is
much cheaper per experiment and can sweep policies and loads that the real system
cannot reach in the available GPU time.

---

Next step: send the reference and this becomes a real document rather than an
outline.
