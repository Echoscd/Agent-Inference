"""
Local (Docker-free) SWE-bench evaluation harness.

Reuses swebench's own TestSpec (build/eval shell scripts) and grading logic,
but rewrites the Docker assumptions (/opt/miniconda3, /testbed, env name
"testbed") to run directly on this machine via a local miniconda.

Per instance:
  1. build_env()   : create a conda env (python + packages) for repo+version
  2. setup_repo()  : git clone + checkout base_commit + pip install repo
  3. evaluate()    : apply candidate patch, run eval_script (applies test_patch
                     + runs FAIL_TO_PASS/PASS_TO_PASS), grade -> resolved

The env + cloned repo live under <work_root>/<instance_id>/ and are reused
across calls so the agent loop can read/run against the same testbed.
"""
import os
import re
import json
import subprocess
import time
from dataclasses import dataclass, field
from typing import Optional

from swebench.harness.test_spec.test_spec import make_test_spec
from swebench.harness.grading import get_logs_eval, get_eval_tests_report, get_resolution_status
from swebench.harness.constants import FAIL_TO_PASS, PASS_TO_PASS
import paths

CONDA_PATH = "/root/miniconda3"
WORK_ROOT  = paths.WORK_ROOT


def _env_name(instance_id: str) -> str:
    return "swb_" + re.sub(r"[^a-zA-Z0-9_]", "_", instance_id)


def _repo_dir(instance_id: str, work_root: str) -> str:
    # NOTE: must not contain the substring "testbed" (we word-replace it below)
    return os.path.join(work_root, instance_id, "src")


def _localize(script: str, repo_dir: str, env_name: str) -> str:
    """Rewrite Docker-image paths/env to local equivalents."""
    s = script.replace("/opt/miniconda3", CONDA_PATH)
    s = s.replace("/testbed", repo_dir)          # working copy dir
    s = re.sub(r"\btestbed\b", env_name, s)      # remaining bare token = conda env name
    return s


def _run(script: str, log_fp: Optional[str] = None, timeout: int = 2400) -> tuple[int, str]:
    """Run a bash script, capture combined output, optionally tee to log_fp."""
    proc = subprocess.run(
        ["bash", "-c", script],
        capture_output=True, text=True, timeout=timeout,
    )
    out = proc.stdout + proc.stderr
    if log_fp:
        with open(log_fp, "w") as f:
            f.write(out)
    return proc.returncode, out


@dataclass
class EvalResult:
    instance_id: str
    resolved: bool
    status: str                       # RESOLVED_FULL / PARTIAL / NO
    report: dict = field(default_factory=dict)
    build_s: float = 0.0
    eval_s: float = 0.0
    error: Optional[str] = None


class Instance:
    """A prepared testbed for one SWE-bench instance (env + cloned repo)."""

    def __init__(self, instance: dict, work_root: str = WORK_ROOT):
        self.instance   = instance
        self.iid        = instance["instance_id"]
        self.spec       = make_test_spec(instance)
        self.work_root  = work_root
        self.repo_dir   = _repo_dir(self.iid, work_root)
        self.env_name   = _env_name(self.iid)
        self.workdir    = os.path.dirname(self.repo_dir)
        os.makedirs(self.workdir, exist_ok=True)

    # ── build phase (slow, run once) ──────────────────────────────────────────
    def build(self, timeout: int = 2400) -> float:
        t0 = time.perf_counter()
        env_sh  = _localize("\n".join(self.spec.env_script_list),  self.repo_dir, self.env_name)
        repo_sh = _localize("\n".join(self.spec.repo_script_list), self.repo_dir, self.env_name)
        # clone into repo_dir; repo_script does `git clone ... <repo_dir>`
        rc, out = _run("set -uxo pipefail\n" + env_sh, timeout=timeout)
        if rc != 0:
            raise RuntimeError(f"[{self.iid}] env build failed:\n{out[-2000:]}")
        rc, out = _run("set -uxo pipefail\n" + repo_sh, timeout=timeout)
        if rc != 0:
            raise RuntimeError(f"[{self.iid}] repo setup failed:\n{out[-2000:]}")
        return time.perf_counter() - t0

    def is_built(self) -> bool:
        return os.path.isdir(os.path.join(self.repo_dir, ".git"))

    # ── apply a candidate patch to the working copy ───────────────────────────
    def apply_patch(self, patch: str) -> tuple[bool, str]:
        if not patch.strip():
            return False, "empty patch"
        pf = os.path.join(self.workdir, "candidate.patch")
        with open(pf, "w") as f:
            f.write(patch if patch.endswith("\n") else patch + "\n")
        # try a few apply strategies (git apply, then patch -p1) like the official harness
        for cmd in [f"cd {self.repo_dir} && git apply -v {pf}",
                    f"cd {self.repo_dir} && git apply -v --3way {pf}",
                    f"cd {self.repo_dir} && patch --batch --fuzz=5 -p1 -i {pf}"]:
            rc, out = _run(cmd, timeout=120)
            if rc == 0:
                return True, "applied"
        return False, out[-1000:]

    def reset(self):
        """Revert working copy to base_commit (drop any applied patch)."""
        _run(f"cd {self.repo_dir} && git checkout -- . && git clean -fdq", timeout=120)

    # ── eval phase ────────────────────────────────────────────────────────────
    def evaluate(self, candidate_patch: str, timeout: int = 2400) -> EvalResult:
        """Apply candidate patch, run tests, grade. Assumes build() already done."""
        res = EvalResult(self.iid, False, "RESOLVED_NO")
        t0 = time.perf_counter()
        self.reset()
        ok, msg = self.apply_patch(candidate_patch)
        if not ok:
            res.error = f"patch apply failed: {msg}"
            res.eval_s = time.perf_counter() - t0
            return res
        eval_sh = _localize(self.spec.eval_script, self.repo_dir, self.env_name)
        log_fp = os.path.join(self.workdir, "eval.log")
        try:
            _run(eval_sh, log_fp=log_fp, timeout=timeout)
        except subprocess.TimeoutExpired:
            res.error = "eval timeout"
            res.eval_s = time.perf_counter() - t0
            return res
        # grade using swebench's own parser + report
        status_map, found = get_logs_eval(self.spec, log_fp)
        if not found:
            res.error = "test output markers not found"
            res.eval_s = time.perf_counter() - t0
            return res
        gold = {
            FAIL_TO_PASS: self.instance[FAIL_TO_PASS] if isinstance(self.instance[FAIL_TO_PASS], list)
                          else json.loads(self.instance[FAIL_TO_PASS]),
            PASS_TO_PASS: self.instance[PASS_TO_PASS] if isinstance(self.instance[PASS_TO_PASS], list)
                          else json.loads(self.instance[PASS_TO_PASS]),
        }
        report = get_eval_tests_report(status_map, gold)
        status = get_resolution_status(report)
        res.report   = report
        res.status   = str(status)
        res.resolved = res.status == "RESOLVED_FULL"
        res.eval_s   = time.perf_counter() - t0
        return res


# ── CLI: validate harness on one instance with the GOLD patch ──────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", default="psf__requests-1142")
    ap.add_argument("--data", default=paths.SWEBENCH_DATA)
    ap.add_argument("--patch", default="gold", help="'gold' to use the reference patch, or a file path")
    args = ap.parse_args()

    inst = None
    for line in open(args.data):
        d = json.loads(line)
        if d["instance_id"] == args.instance_id:
            inst = d
            break
    if inst is None:
        raise SystemExit(f"instance {args.instance_id} not found")

    patch = inst["patch"] if args.patch == "gold" else open(args.patch).read()

    obj = Instance(inst)
    print(f"[{obj.iid}] repo={inst['repo']} version={inst['version']} env={obj.env_name}")
    print(f"[{obj.iid}] testbed dir: {obj.repo_dir}")
    if not obj.is_built():
        print(f"[{obj.iid}] building env + repo ...")
        bt = obj.build()
        print(f"[{obj.iid}] build done in {bt:.1f}s")
    else:
        print(f"[{obj.iid}] testbed already built, reusing")

    print(f"[{obj.iid}] evaluating with {args.patch} patch ...")
    r = obj.evaluate(patch)
    print(f"\n{'='*60}")
    print(f"  instance: {r.instance_id}")
    print(f"  resolved: {r.resolved}   status: {r.status}")
    print(f"  eval_s:   {r.eval_s:.1f}s")
    if r.error:
        print(f"  error:    {r.error}")
    if r.report:
        for tid, sub in r.report.items():
            for k, v in sub.items():
                if isinstance(v, list) and v:
                    print(f"    {k}: {len(v)} tests")
    print(f"{'='*60}")
