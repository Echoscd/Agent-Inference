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


def conda_root():
    """Base of the conda install used to build SWE-bench testbeds.

    SWE-bench's setup scripts assume the Docker image layout (/opt/miniconda3);
    the harness rewrites that to whatever conda is actually present here. Set
    AGENT_EXP_CONDA to pin it, otherwise the usual install locations are probed
    and finally `conda info --base` is asked.
    """
    env = os.environ.get("AGENT_EXP_CONDA")
    if env:
        return env
    for c in ("/root/miniconda3", "/opt/miniconda3", "/opt/conda",
              os.path.expanduser("~/miniconda3"), os.path.expanduser("~/anaconda3")):
        if os.path.isdir(os.path.join(c, "bin")):
            return c
    base = os.environ.get("CONDA_PREFIX_1") or os.environ.get("CONDA_PREFIX")
    if base:
        return base
    import shutil
    exe = shutil.which("conda")
    if exe:
        return os.path.dirname(os.path.dirname(os.path.realpath(exe)))
    raise RuntimeError(
        "no conda install found -- SWE-bench testbeds need one. "
        "Install miniconda or set AGENT_EXP_CONDA=/path/to/conda")


def result(*parts):
    """Path inside result/, creating the parent experiment folder."""
    p = os.path.join(RESULT_DIR, *parts)
    os.makedirs(os.path.dirname(p) if os.path.splitext(p)[1] else p, exist_ok=True)
    return p
