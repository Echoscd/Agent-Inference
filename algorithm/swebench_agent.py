"""
SWE-bench agent loop (mini-swe-agent style) for inference-acceleration experiments.

Each turn:
  1. model streams ONE bash command  -> capture TTFT / decode / gen tokens
  2. we execute it in the instance's conda env + cloned testbed (REAL wait)
  3. stdout/stderr fed back as the next observation (context grows)
Loop ends when the model submits or MAX_TURNS is hit.

The realistic *turn count* and inter-turn *wait* (real test runs / file reads)
are what make this a representative serving workload, not a single-shot prompt.

After the loop we extract `git diff` from the testbed and grade it with the
local (Docker-free) harness -> resolved? + full per-turn timing trace.
"""
import re
import os
import json
import time
import subprocess
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from swebench_local_harness import Instance, CONDA_PATH
import paths

# ── Config ──────────────────────────────────────────────────────────────────
BASE_URL   = "http://localhost:8000/v1"
MODEL      = os.environ.get("AGENT_MODEL", "qwen3-32b")  # --served-model-name of the
                          # running vLLM; every launcher overrides it via --model
MAX_TURNS  = 20
MAX_TOKENS = 8192                 # per-turn generation cap (set high per experiment policy)
BASH_TIMEOUT = 90                 # seconds per command (real inter-turn wait, bounded)

client = OpenAI(base_url=BASE_URL, api_key="EMPTY")

SUBMIT_SENTINEL = "SWEBENCH_TASK_SUBMIT"

SYSTEM_PROMPT = f"""\
You are an autonomous software engineer fixing a bug in a real Python repository.
The repository is already checked out at the working directory; your shell starts there.

Work in a strict loop. Each turn output EXACTLY ONE bash command inside a single
```bash ... ``` fenced block, and nothing else. The system executes it and returns
the combined stdout/stderr as the next Observation. Use it to explore, read files,
make edits, and run tests.

Guidelines:
- Investigate efficiently. Spend at most a few turns locating the code; do NOT
  repeat near-identical grep commands. Use `cat -n <file>` to read a file with
  line numbers before editing it.
- As soon as you have located the relevant code, MAKE THE EDIT. Do not keep
  exploring. A run that never edits a source file cannot fix the bug.
- Make minimal, correct edits to the SOURCE files (use python to rewrite a file,
  `sed -i`, or a heredoc). Verify your edit took effect with `git diff`.
- Do NOT edit test files; they are reset before grading.
- You may run the project's tests to check your fix.
- When you are confident the bug is fixed, output exactly this command to finish:
  ```bash
  echo {SUBMIT_SENTINEL}
  ```

Output ONLY the single ```bash``` block each turn."""

_BASH_RE = re.compile(r"```bash\s*(.*?)```", re.DOTALL)


def extract_bash(text: str) -> Optional[str]:
    m = _BASH_RE.findall(text)
    return m[-1].strip() if m else None


# ── per-turn timing ───────────────────────────────────────────────────────────
@dataclass
class TurnTiming:
    ttft_s: float
    decode_s: float
    total_s: float
    prompt_tokens: int
    gen_tokens: int
    wait_s: float = 0.0           # real tool-execution wall time after generation

    @property
    def gen_tps(self) -> float:
        return self.gen_tokens / self.decode_s if self.decode_s > 0 else 0.0


def _stream_call(messages: list[dict]) -> tuple[str, TurnTiming]:
    t0 = time.perf_counter()
    ttft = None
    parts = []
    ptok = gtok = 0
    stream = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.0,
        max_tokens=MAX_TOKENS, stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(chunk.choices[0].delta.content)
        if getattr(chunk, "usage", None) is not None:
            ptok = chunk.usage.prompt_tokens or 0
            gtok = chunk.usage.completion_tokens or 0
    total = time.perf_counter() - t0
    ttft = ttft or total
    return "".join(parts).strip(), TurnTiming(ttft, max(total - ttft, 0.0), total, ptok, gtok)


@dataclass
class AgentResult:
    instance_id: str
    resolved: bool = False
    status: str = "RESOLVED_NO"
    turns: int = 0
    submitted: bool = False
    timings: list[TurnTiming] = field(default_factory=list)
    patch: str = ""
    build_s: float = 0.0
    eval_s: float = 0.0
    error: Optional[str] = None

    # aggregate inference metrics
    @property
    def total_gen_tokens(self) -> int:    return sum(t.gen_tokens for t in self.timings)
    @property
    def total_prompt_tokens(self) -> int: return sum(t.prompt_tokens for t in self.timings)
    @property
    def total_decode_s(self) -> float:    return sum(t.decode_s for t in self.timings)
    @property
    def total_ttft_s(self) -> float:      return sum(t.ttft_s for t in self.timings)
    @property
    def total_wait_s(self) -> float:      return sum(t.wait_s for t in self.timings)
    @property
    def gen_tps(self) -> float:
        d = self.total_decode_s
        return self.total_gen_tokens / d if d > 0 else 0.0


def _run_in_testbed(obj: Instance, cmd: str, timeout: int) -> str:
    """Execute a bash command inside the instance's conda env + repo dir. Real wait."""
    wrapped = (
        f"source {CONDA_PATH}/bin/activate && conda activate {obj.env_name} 2>/dev/null; "
        f"cd {obj.repo_dir} && ({cmd})"
    )
    try:
        p = subprocess.run(["bash", "-c", wrapped], capture_output=True, text=True, timeout=timeout)
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        out = f"[command timed out after {timeout}s]"
    if len(out) > 4000:
        out = out[:2000] + "\n...[truncated]...\n" + out[-2000:]
    return out.strip() or "[no output]"


def run_agent(instance: dict, work_root=None, verbose=True, max_turns=MAX_TURNS) -> AgentResult:
    obj = Instance(instance) if work_root is None else Instance(instance, work_root)
    res = AgentResult(obj.iid)

    # build testbed (env + clone) once
    if not obj.is_built():
        if verbose: print(f"[{obj.iid}] building testbed ...")
        try:
            res.build_s = obj.build()
        except Exception as e:
            res.error = f"build failed: {e}"
            return res
    obj.reset()  # clean working copy before the agent touches it

    problem = instance["problem_statement"]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"Repository: {instance['repo']} (working dir already checked out at the bug commit).\n\n"
            f"Bug report / issue to fix:\n\n{problem}\n\n"
            f"Begin by investigating the codebase. Output your first bash command."},
    ]

    for turn in range(1, max_turns + 1):
        try:
            assistant, timing = _stream_call(messages)
        except Exception as e:
            res.error = f"model call failed at turn {turn}: {e}"
            break
        messages.append({"role": "assistant", "content": assistant})
        cmd = extract_bash(assistant)

        if verbose:
            print(f"  ── turn {turn} | TTFT={timing.ttft_s*1000:.0f}ms "
                  f"decode={timing.decode_s:.1f}s gen={timing.gen_tokens}tok "
                  f"({timing.gen_tps:.0f}t/s)")
            print(f"     $ {(cmd or '<no command>')[:120]}")

        if cmd is None:
            obs = "Observation: No ```bash``` block found. Output exactly one bash command."
            messages.append({"role": "user", "content": obs})
            res.timings.append(timing)
            continue

        if cmd.strip() == f"echo {SUBMIT_SENTINEL}" or SUBMIT_SENTINEL in cmd:
            res.submitted = True
            res.timings.append(timing)
            res.turns = turn
            if verbose: print(f"     -> submitted")
            break

        # real tool execution = real inter-turn wait
        w0 = time.perf_counter()
        obs = _run_in_testbed(obj, cmd, BASH_TIMEOUT)
        timing.wait_s = time.perf_counter() - w0
        res.timings.append(timing)
        messages.append({"role": "user", "content": f"Observation:\n{obs}"})
        res.turns = turn

    # extract the agent's patch from the working copy, then grade cleanly
    diff = subprocess.run(
        ["bash", "-c", f"cd {obj.repo_dir} && git -c core.fileMode=false diff"],
        capture_output=True, text=True, timeout=120,
    ).stdout
    res.patch = diff

    if diff.strip():
        ev = obj.evaluate(diff)
        res.resolved = ev.resolved
        res.status   = ev.status
        res.eval_s   = ev.eval_s
        if ev.error and not res.error:
            res.error = ev.error
    else:
        res.status = "NO_PATCH"

    return res


# ── CLI: run the agent on one instance ─────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", default="psf__requests-1142")
    ap.add_argument("--data", default=paths.SWEBENCH_DATA)
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    args = ap.parse_args()

    inst = next((json.loads(l) for l in open(args.data)
                 if json.loads(l)["instance_id"] == args.instance_id), None)
    if inst is None:
        raise SystemExit(f"instance {args.instance_id} not found")

    print(f"=== agent on {args.instance_id} ({inst['repo']} {inst['version']}) ===")
    r = run_agent(inst, max_turns=args.max_turns)
    print(f"\n{'='*60}")
    print(f"  resolved:   {r.resolved}  ({r.status})")
    print(f"  turns:      {r.turns}  | submitted: {r.submitted}")
    print(f"  gen tokens: {r.total_gen_tokens}  | prompt tokens: {r.total_prompt_tokens}")
    print(f"  decode:     {r.total_decode_s:.1f}s  | gen speed: {r.gen_tps:.0f} t/s")
    print(f"  tool wait:  {r.total_wait_s:.1f}s  (real inter-turn execution)")
    print(f"  build:      {r.build_s:.1f}s | eval: {r.eval_s:.1f}s")
    if r.error: print(f"  error:      {r.error}")
    print(f"{'='*60}")
