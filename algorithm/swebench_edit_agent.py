"""
Edit-style SWE-bench agent: "long reasoning + file edits" workload.

Unlike the minimal one-bash-line-per-turn agent, here each turn the model emits:
  1. a Reasoning: block (analysis -> elicits long generation), then
  2. exactly ONE action:
       OPEN: <path>                 -> returns the file with line numbers
       EDIT: <path> + SEARCH/REPLACE blocks (Aider style, full code reproduced)
       RUN:  <shell cmd>            -> run tests/commands in the conda env
       SUBMIT                       -> finish

This makes decode (not tool execution) dominate per turn, matching real coding
agents, so the server actually does sustained generation work.

Reuses Instance (harness) + TurnTiming/AgentResult (swebench_agent).
"""
import re
import os
import json
import time
import subprocess
from typing import Optional

from openai import OpenAI

from swebench_local_harness import Instance, CONDA_PATH
from swebench_agent import TurnTiming, AgentResult, BASE_URL, MODEL, BASH_TIMEOUT
import ab_tape   # A/B experiment record/replay tape (env-gated; no-op otherwise)

MAX_TURNS  = 20
MAX_TOKENS = 8192          # high per-turn cap so long reasoning + edits aren't truncated
OPEN_MAX_LINES = 400       # cap file view size

import os as _os
import urllib.request as _urlreq
import paths
BASE_URL = _os.environ.get("AGENT_BASE_URL", BASE_URL)   # point at ThunderAgent proxy when set
client = OpenAI(base_url=BASE_URL, api_key="EMPTY")

def _release_program(pid: str):
    """Tell ThunderAgent the program is done (frees its scheduling slot). Best-effort."""
    base = BASE_URL.rsplit("/v1", 1)[0]
    try:
        req = _urlreq.Request(base + "/programs/release",
                              data=json.dumps({"program_id": pid}).encode(),
                              headers={"Content-Type": "application/json"}, method="POST")
        _urlreq.urlopen(req, timeout=5).read()
    except Exception:
        pass

SUBMIT_SENTINEL = "SUBMIT"

SYSTEM_PROMPT = """\
You are an expert software engineer fixing a bug in a real Python repository that
is checked out in the working directory.

Work in a loop. EACH turn you MUST output, in this exact order:

1. A `Reasoning:` section: think step by step — what the bug is, where it lives,
   what the fix should be, and why. Be thorough and explicit.

2. EXACTLY ONE action, as the LAST thing in your message, in one of these forms:

   OPEN: path/to/file.py
       (the system returns that file with line numbers)

   RUN: <shell command>
       (e.g. run the failing test; the system returns its output)

   EDIT: path/to/file.py
   ```
   <<<<<<< SEARCH
   <exact existing lines to find, copied verbatim>
   =======
   <the replacement lines>
   >>>>>>> REPLACE
   ```
   (you may include multiple SEARCH/REPLACE blocks under one EDIT; the SEARCH text
    must match the current file EXACTLY, including indentation)

   SUBMIT
       (only when you are confident the bug is fixed)

Rules:
- Open and read the relevant file(s) before editing them.
- Edit only SOURCE files, never test files (tests are reset before grading).
- Make minimal, correct changes. Reproduce existing code exactly in SEARCH blocks.
- Prefer to RUN the failing test after editing to confirm the fix, then SUBMIT.
"""

# ── action parsing ──────────────────────────────────────────────────────────
_OPEN_RE = re.compile(r"OPEN:\s*(\S+)")
_RUN_RE  = re.compile(r"RUN:\s*(.+)")
_EDIT_RE = re.compile(r"EDIT:\s*(\S+)")
_SR_RE   = re.compile(r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE", re.DOTALL)


def parse_action(text: str) -> tuple[str, dict]:
    """Return (kind, payload). kind in {open, run, edit, submit, none}.

    A real action (edit/open/run) takes priority over SUBMIT: models often write
    an EDIT and then say SUBMIT in the same message — we must apply the edit, not
    drop it. SUBMIT only terminates when the message contains no other action.
    """
    if _EDIT_RE.search(text):
        m = _EDIT_RE.search(text)
        blocks = _SR_RE.findall(text)
        return "edit", {"path": m.group(1).strip(), "blocks": blocks}
    if _OPEN_RE.search(text):
        path = _OPEN_RE.findall(text)[-1].strip()
        return "open", {"path": path}
    if _RUN_RE.search(text):
        cmd = _RUN_RE.findall(text)[-1].strip().strip("`")
        return "run", {"cmd": cmd}
    # only a bare SUBMIT (no edit/open/run in the message) ends the run
    tail = text.strip().splitlines()[-5:] if text.strip() else []
    if any(l.strip() == SUBMIT_SENTINEL for l in tail):
        return "submit", {}
    return "none", {}


# ── tool implementations (operate on the testbed working copy) ────────────────
def tool_open(obj: Instance, path: str) -> str:
    fp = os.path.join(obj.repo_dir, path)
    if not os.path.isfile(fp):
        # try to locate by basename
        base = os.path.basename(path)
        hits = subprocess.run(["bash", "-c", f"cd {obj.repo_dir} && find . -name {base!r} | head -5"],
                              capture_output=True, text=True).stdout.strip()
        return f"File not found: {path}\nClosest matches:\n{hits or '(none)'}"
    with open(fp, errors="replace") as f:
        lines = f.read().splitlines()
    shown = lines[:OPEN_MAX_LINES]
    body = "\n".join(f"{i+1:5d}  {l}" for i, l in enumerate(shown))
    more = "" if len(lines) <= OPEN_MAX_LINES else f"\n... ({len(lines)-OPEN_MAX_LINES} more lines)"
    return f"{path} ({len(lines)} lines):\n{body}{more}"


def _apply_one(content: str, search: str, replace: str) -> Optional[str]:
    """Apply a SEARCH/REPLACE block. Exact match first, then whitespace-tolerant
    (match by stripped lines, re-indent REPLACE to the file's actual indentation).
    Returns new content, or None if the search block can't be located."""
    if search in content:
        return content.replace(search, replace, 1)
    # whitespace-tolerant: compare lines with leading/trailing space stripped
    flines = content.splitlines()
    slines = [s.rstrip() for s in search.splitlines()]
    sstrip = [s.strip() for s in slines if s.strip() != ""]
    if not sstrip:
        return None
    for i in range(len(flines) - len(sstrip) + 1):
        window = [flines[i + j].strip() for j in range(len(sstrip))]
        if window == sstrip:
            # indentation offset = leading ws of first matched file line vs first search line
            file_indent = flines[i][:len(flines[i]) - len(flines[i].lstrip())]
            s_first = next(s for s in search.splitlines() if s.strip())
            s_indent = s_first[:len(s_first) - len(s_first.lstrip())]
            # re-indent replace block by (file_indent - s_indent)
            rep_lines = replace.splitlines()
            new_rep = []
            for rl in rep_lines:
                if rl.startswith(s_indent):
                    new_rep.append(file_indent + rl[len(s_indent):])
                else:
                    new_rep.append(rl)
            new_flines = flines[:i] + new_rep + flines[i + len(sstrip):]
            return "\n".join(new_flines) + ("\n" if content.endswith("\n") else "")
    return None


def tool_edit(obj: Instance, path: str, blocks: list[tuple[str, str]]) -> str:
    fp = os.path.join(obj.repo_dir, path)
    if not os.path.isfile(fp):
        return f"Cannot edit, file not found: {path}"
    if not blocks:
        return "No valid SEARCH/REPLACE block found. Use the exact <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE format."
    with open(fp, errors="replace") as f:
        content = f.read()
    applied = 0
    for search, replace in blocks:
        if search == "":
            continue
        new = _apply_one(content, search, replace)
        if new is None:
            return (f"SEARCH block not found in {path} (tried exact + whitespace-tolerant match). "
                    f"Failed block starts with:\n{search.splitlines()[0] if search.splitlines() else ''}")
        content = new
        applied += 1
    with open(fp, "w") as f:
        f.write(content)
    return f"Applied {applied} edit(s) to {path}. Run the tests to verify."


def tool_run(obj: Instance, cmd: str) -> str:
    wrapped = (f"source {CONDA_PATH}/bin/activate && conda activate {obj.env_name} 2>/dev/null; "
               f"cd {obj.repo_dir} && ({cmd})")
    try:
        p = subprocess.run(["bash", "-c", wrapped], capture_output=True, text=True, timeout=BASH_TIMEOUT)
        out = (p.stdout or "") + (p.stderr or "")
    except subprocess.TimeoutExpired:
        out = f"[timed out after {BASH_TIMEOUT}s]"
    if len(out) > 4000:
        out = out[:2000] + "\n...[truncated]...\n" + out[-2000:]
    return out.strip() or "[no output]"


# ── streaming call (same timing capture as the bash agent) ────────────────────
def _stream_call(messages: list[dict], max_tokens: int = None, program_id: str = None,
                 turn: int = None) -> tuple[str, TurnTiming]:
    t0 = time.perf_counter(); t_start = ab_tape.now(); ttft = None; parts = []; ptok = gtok = 0
    # X-Session-ID header = ThunderAgent program_id (ignored by plain vLLM)
    extra_headers = {"X-Session-ID": program_id} if program_id else {}
    # A/B pass 2: tell ThunderAgent this call's KNOWN decode length (from pass-1 tape)
    # so the density policy can score by the real upcoming decode (ignored by plain vLLM).
    if program_id is not None and turn is not None:
        kd = ab_tape.known_decode(program_id, turn)
        if kd:
            extra_headers["X-Decode-Len"] = str(int(kd))
    stream = client.chat.completions.create(
        model=MODEL, messages=messages, temperature=0.0,
        max_tokens=max_tokens or MAX_TOKENS, stream=True, stream_options={"include_usage": True},
        extra_headers=extra_headers or None,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(chunk.choices[0].delta.content)
        if getattr(chunk, "usage", None) is not None:
            ptok = chunk.usage.prompt_tokens or 0
            gtok = chunk.usage.completion_tokens or 0
    total = time.perf_counter() - t0; ttft = ttft or total
    text = "".join(parts).strip()
    # A/B pass(es): save full prefill (messages) + decode (text) content + timing for this call.
    # wait_s = TTFT (pause+queue+prefill), decode_s = generation time. tool_wait is set
    # after the tool runs (in run_agent), so it's left 0 here — the green sliver in the Gantt.
    if program_id is not None and turn is not None:
        ab_tape.record_call(program_id, turn, messages, text, ptok, gtok,
                            wait_s=ttft, decode_s=max(total - ttft, 0.0),
                            t_start_s=t_start)
    return text, TurnTiming(ttft, max(total - ttft, 0.0), total, ptok, gtok)


def run_agent(instance: dict, work_root=None, verbose=True, max_turns=MAX_TURNS,
              grow_to_max=False, max_context=40960) -> AgentResult:
    """If grow_to_max: keep taking turns (ignoring SUBMIT) until the context fills
    ~max_context tokens, dynamically clamping per-turn max_tokens to avoid overflow.
    Used to drive long-context behaviour up to the model's max length."""
    obj = Instance(instance) if work_root is None else Instance(instance, work_root)
    res = AgentResult(obj.iid)
    if grow_to_max:
        max_turns = max(max_turns, 1000)            # effectively uncapped; context size is the stop
    last_prompt_tokens = 0                          # tracks growing context for clamping/stop
    if not obj.is_built():
        if verbose: print(f"[{obj.iid}] building testbed ...")
        try:
            res.build_s = obj.build()
        except Exception as e:
            res.error = f"build failed: {e}"; return res
    obj.reset()

    # seed with repo file tree so the model can navigate without grep-spam
    tree = subprocess.run(
        ["bash", "-c", f"cd {obj.repo_dir} && git ls-files | grep -E '\\.py$' | head -200"],
        capture_output=True, text=True).stdout.strip()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content":
            f"Repository: {instance['repo']} (checked out at the bug commit).\n\n"
            f"Bug report:\n\n{instance['problem_statement']}\n\n"
            f"Python files in the repo (first 200):\n{tree}\n\n"
            f"Begin. Reason about the bug, then OPEN the most relevant file."},
    ]

    for turn in range(1, max_turns + 1):
        # clamp generation so prompt + gen never exceeds the model's max length
        # (applies whether or not grow_to_max: protects unbounded-turn runs too)
        budget = max_context - last_prompt_tokens - 1024          # margin for next observation
        if budget < 512:                                          # no room for another turn
            if verbose: print(f"     -> context full ({last_prompt_tokens} tok), stopping")
            break
        cur_max_tokens = min(MAX_TOKENS, budget)
        try:
            assistant, timing = _stream_call(messages, max_tokens=cur_max_tokens, program_id=obj.iid, turn=turn)
        except Exception as e:
            res.error = f"model call failed at turn {turn}: {e}"; break
        last_prompt_tokens = timing.prompt_tokens
        messages.append({"role": "assistant", "content": assistant})
        kind, payload = parse_action(assistant)

        if verbose:
            print(f"  ── turn {turn} | {kind:6s} | prefill={timing.ttft_s*1000:.0f}ms "
                  f"gen={timing.gen_tokens}tok decode={timing.decode_s:.1f}s ({timing.gen_tps:.0f}t/s) "
                  f"prompt={timing.prompt_tokens}"
                  + (f"  [ctx {timing.prompt_tokens}/{max_context}]" if grow_to_max else ""))

        if kind == "submit":
            if grow_to_max:
                # ignore submit; force the agent to keep working so context keeps growing
                res.timings.append(timing); res.turns = turn
                messages.append({"role": "user", "content":
                    "Do not submit yet. Continue improving: OPEN another relevant file, "
                    "review it thoroughly, and refine the fix. Keep investigating."})
                continue
            res.submitted = True; res.timings.append(timing); res.turns = turn
            if verbose: print("     -> SUBMIT");
            break

        w0 = time.perf_counter()
        if kind == "open":
            obs = tool_open(obj, payload["path"])
        elif kind == "edit":
            obs = tool_edit(obj, payload["path"], payload["blocks"])
        elif kind == "run":
            obs = tool_run(obj, payload["cmd"])
        else:
            obs = ("No valid action found. End your message with exactly one of: "
                   "OPEN: <file> / RUN: <cmd> / EDIT: <file> + SEARCH/REPLACE / SUBMIT.")
        timing.wait_s = time.perf_counter() - w0
        res.timings.append(timing)
        res.turns = turn
        messages.append({"role": "user", "content": f"Observation:\n{obs}"})

    diff = subprocess.run(["bash", "-c", f"cd {obj.repo_dir} && git -c core.fileMode=false diff"],
                          capture_output=True, text=True, timeout=120).stdout
    _release_program(obj.iid)   # free ThunderAgent scheduling slot (no-op vs plain vLLM)
    res.patch = diff
    if diff.strip():
        ev = obj.evaluate(diff)
        res.resolved, res.status, res.eval_s = ev.resolved, ev.status, ev.eval_s
        if ev.error and not res.error:
            res.error = ev.error
    else:
        res.status = "NO_PATCH"
    return res


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", default="psf__requests-1142")
    ap.add_argument("--data", default=paths.SWEBENCH_DATA)
    ap.add_argument("--max-turns", type=int, default=MAX_TURNS)
    ap.add_argument("--model", default=None)
    ap.add_argument("--grow-to-max", action="store_true",
                    help="keep taking turns (ignore SUBMIT) until context fills max-context")
    ap.add_argument("--max-context", type=int, default=40960)
    args = ap.parse_args()
    if args.model:
        MODEL = args.model
    inst = next((json.loads(l) for l in open(args.data)
                 if json.loads(l)["instance_id"] == args.instance_id), None)
    if inst is None:
        raise SystemExit(f"instance {args.instance_id} not found")
    print(f"=== edit-agent on {args.instance_id} ({inst['repo']} {inst['version']}) "
          f"grow_to_max={args.grow_to_max} ===")
    r = run_agent(inst, max_turns=args.max_turns,
                  grow_to_max=args.grow_to_max, max_context=args.max_context)
    print(f"\n{'='*60}")
    print(f"  resolved:   {r.resolved}  ({r.status})")
    print(f"  turns:      {r.turns}  submitted: {r.submitted}")
    print(f"  gen tokens: {r.total_gen_tokens}  | decode: {r.total_decode_s:.1f}s "
          f"| gen speed: {r.gen_tps:.0f} t/s")
    print(f"  TOTAL prefill: {r.total_ttft_s:.1f}s   TOTAL decode: {r.total_decode_s:.1f}s   "
          f"prefill/(prefill+decode) = {r.total_ttft_s/(r.total_ttft_s+r.total_decode_s)*100:.1f}%")
    print(f"  tool wait:  {r.total_wait_s:.1f}s")
    if r.error: print(f"  error:      {r.error}")
    print(f"{'='*60}")
