"""Repo-relative paths. Nothing in this project hardcodes an absolute path.

REPO_ROOT is the directory containing algorithm/ -- resolved from this file's
own location, so scripts work from any cwd and from any clone location.
Every default can still be overridden by an environment variable, which is what
the shell harness uses when it wants results somewhere else.
"""
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env(name, *parts):
    return os.environ.get(name) or os.path.join(REPO_ROOT, *parts)


ALGORITHM = os.path.join(REPO_ROOT, "algorithm")
DATA_DIR = _env("AGENT_EXP_DATA", "data")
RESULT_DIR = _env("AGENT_EXP_RESULT", "result")
# SWE-bench repo checkouts: big and rebuildable, so overridable to another disk
WORK_ROOT = _env("AGENT_EXP_WORK_ROOT", "swebench_runs")

SWEBENCH_DATA = os.path.join(DATA_DIR, "swebench_verified.jsonl")
HUMANEVAL_DATA = os.path.join(DATA_DIR, "humaneval.jsonl")
IDS80 = os.path.join(DATA_DIR, "ids80.txt")


def result(*parts):
    """Path inside result/, creating the parent experiment folder."""
    p = os.path.join(RESULT_DIR, *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p
