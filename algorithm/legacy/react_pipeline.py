"""
ReAct pipeline for HumanEval: Write code -> Run -> Feedback -> Repeat
"""
import re
import json
import subprocess
import sys
import tempfile
import os
import time
from openai import OpenAI
from dataclasses import dataclass, field
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL  = "http://localhost:8000/v1"
MODEL     = "qwen3-4b"
MAX_ITERS = 5
TIMEOUT   = 10  # seconds per code execution

client = OpenAI(base_url=BASE_URL, api_key="EMPTY")

# ── Prompts ───────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert Python programmer. You solve coding problems using a strict ReAct loop.

Each turn you MUST produce:
1. A **Thought** block: brief reasoning about the problem or the error.
2. A **Code** block: the complete Python function(s) in a ```python ... ``` fence.

Do NOT produce anything else. Do NOT call the function yourself.
The system will execute your code and return test results as an Observation.
Keep iterating until all tests pass.

Format exactly:
Thought: <your reasoning>
```python
<complete function implementation>
```
"""

def make_user_prompt(prompt: str) -> str:
    return f"Solve the following Python programming problem:\n\n{prompt}"

def make_observation(passed: int, total: int, error: Optional[str]) -> str:
    if error is None:
        return f"Observation: All {total} tests passed. ✓"
    return (
        f"Observation: {passed}/{total} tests passed.\n"
        f"Error:\n{error}"
    )

# ── Code extraction ───────────────────────────────────────────────────────────
_CODE_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)

def extract_code(text: str) -> Optional[str]:
    matches = _CODE_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()

# ── Code execution ────────────────────────────────────────────────────────────
def run_tests(code: str, test_code: str, entry_point: str) -> tuple[int, int, Optional[str]]:
    """Returns (passed, total, error_message). error_message is None if all tests pass."""
    full_code = code + "\n\n" + test_code + f"\n\ncheck({entry_point!r})\n"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(full_code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=TIMEOUT
        )
        if result.returncode == 0:
            return 1, 1, None
        else:
            err = (result.stderr or result.stdout).strip()
            if len(err) > 800:
                err = err[-800:]
            return 0, 1, err
    except subprocess.TimeoutExpired:
        return 0, 1, f"Timeout after {TIMEOUT}s"
    finally:
        os.unlink(fname)

# ── Per-request timing ────────────────────────────────────────────────────────
@dataclass
class RequestTiming:
    ttft_s: float        # time-to-first-token  ≈ prefill latency (client-side)
    decode_s: float      # time from first to last token ≈ decode latency
    total_s: float       # end-to-end wall time
    prompt_tokens: int
    gen_tokens: int

    @property
    def gen_tps(self) -> float:
        return self.gen_tokens / self.decode_s if self.decode_s > 0 else 0.0


def _call_model_streaming(messages: list[dict]) -> tuple[str, RequestTiming]:
    """Call model with streaming; return (text, timing)."""
    t0 = time.perf_counter()
    ttft: Optional[float] = None
    parts: list[str] = []
    prompt_tokens = 0
    gen_tokens = 0

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=1024,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(chunk.choices[0].delta.content)
        if getattr(chunk, "usage", None) is not None:
            prompt_tokens = chunk.usage.prompt_tokens or 0
            gen_tokens    = chunk.usage.completion_tokens or 0

    total_s = time.perf_counter() - t0
    ttft    = ttft or total_s  # fallback if model returned empty
    decode_s = max(total_s - ttft, 0.0)

    timing = RequestTiming(
        ttft_s=ttft,
        decode_s=decode_s,
        total_s=total_s,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
    )
    return "".join(parts).strip(), timing


def _call_model_basic(messages: list[dict]) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=1024,
    )
    return response.choices[0].message.content.strip()

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class StepRecord:
    iteration: int
    thought: str
    code: str
    passed: int
    total: int
    error: Optional[str]
    timing: Optional[RequestTiming] = None   # populated when timed=True


@dataclass
class SolveResult:
    task_id: str
    success: bool
    iterations: int
    steps: list[StepRecord] = field(default_factory=list)
    final_code: Optional[str] = None

    # Aggregate timing across all iterations (only when timed=True)
    @property
    def total_ttft_s(self) -> float:
        return sum(s.timing.ttft_s for s in self.steps if s.timing)

    @property
    def total_decode_s(self) -> float:
        return sum(s.timing.decode_s for s in self.steps if s.timing)

    @property
    def total_prompt_tokens(self) -> int:
        return sum(s.timing.prompt_tokens for s in self.steps if s.timing)

    @property
    def total_gen_tokens(self) -> int:
        return sum(s.timing.gen_tokens for s in self.steps if s.timing)

# ── ReAct loop ────────────────────────────────────────────────────────────────
def solve(task: dict, verbose: bool = True, timed: bool = False) -> SolveResult:
    task_id     = task["task_id"]
    prompt      = task["prompt"]
    test_code   = task["test"]
    entry_point = task["entry_point"]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": make_user_prompt(prompt)},
    ]

    steps: list[StepRecord] = []
    final_code = None

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_id}  |  entry_point: {entry_point}")
        print(f"{'='*60}")

    for i in range(1, MAX_ITERS + 1):
        if verbose:
            print(f"\n── Iteration {i} ──")

        # ① Call model
        timing: Optional[RequestTiming] = None
        if timed:
            assistant_msg, timing = _call_model_streaming(messages)
            if verbose:
                print(f"   [timing] TTFT={timing.ttft_s*1000:.0f}ms  "
                      f"decode={timing.decode_s:.2f}s  "
                      f"gen={timing.gen_tokens}tok  "
                      f"speed={timing.gen_tps:.1f}tok/s")
        else:
            assistant_msg = _call_model_basic(messages)

        messages.append({"role": "assistant", "content": assistant_msg})

        if verbose:
            print(assistant_msg[:600] + ("..." if len(assistant_msg) > 600 else ""))

        # ② Extract code
        code    = extract_code(assistant_msg)
        thought = assistant_msg.split("```")[0].replace("Thought:", "").strip()

        if code is None:
            obs = "Observation: No ```python``` code block found. Please provide a complete implementation."
            if verbose:
                print(obs)
            messages.append({"role": "user", "content": obs})
            steps.append(StepRecord(i, thought, "", 0, 1, "No code block", timing))
            continue

        final_code = code

        # ③ Run tests
        passed, total, error = run_tests(code, test_code, entry_point)
        obs = make_observation(passed, total, error)

        if verbose:
            print(f"\n{obs}")

        steps.append(StepRecord(i, thought, code, passed, total, error, timing))

        if error is None:
            return SolveResult(task_id, True, i, steps, final_code)

        # ④ Feed observation back
        messages.append({"role": "user", "content": obs})

    return SolveResult(task_id, False, MAX_ITERS, steps, final_code)

# ── CLI entry ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ReAct pipeline on a single HumanEval problem")
    parser.add_argument("--task-id", default="HumanEval/0")
    parser.add_argument("--data",    default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                             os.path.abspath(__file__)))), "data", "humaneval.jsonl"))
    parser.add_argument("--timed",   action="store_true",
                        help="Use streaming to measure per-iteration TTFT and decode time")
    args = parser.parse_args()

    tasks = {}
    with open(args.data) as f:
        for line in f:
            t = json.loads(line)
            tasks[t["task_id"]] = t

    if args.task_id not in tasks:
        print(f"Task {args.task_id!r} not found. Available: {list(tasks.keys())[:5]} ...")
        sys.exit(1)

    result = solve(tasks[args.task_id], verbose=True, timed=args.timed)

    print(f"\n{'='*60}")
    print(f"Result: {'PASS ✓' if result.success else 'FAIL ✗'} in {result.iterations} iteration(s)")
    if args.timed and result.steps:
        print(f"Total TTFT:   {result.total_ttft_s*1000:.0f} ms")
        print(f"Total decode: {result.total_decode_s:.2f} s")
        print(f"Prompt tok:   {result.total_prompt_tokens}")
        print(f"Gen tok:      {result.total_gen_tokens}")
