"""make_policy(name, ...) -> SchedulingPolicy. Add a new algorithm here (one line)
after creating its scheduling/<name>.py file."""
from __future__ import annotations

from .base import SchedulingPolicy
from .size import SizePolicy
from .density import DensityPolicy
from .dual_descent import DualDescentPolicy
from .fidelity import FidelityPolicy
from .hazard_grade import HazardGradePolicy
from .hazard_grade_v2 import HazardGradeV2Policy

POLICY_NAMES = ["size", "density", "dual_descent", "fidelity", "hazard_grade",
                "hazard_grade_v2"]


def make_policy(
    name: str,
    *,
    alpha: float,
    decode_hat: float,
    dd_eta0: float,
    # hazard_grade-only knobs (ignored by the other policies)
    hz_decode_mean: float = 1650.0,
    hz_prompt_mean: float = 1800.0,
    hz_decode_reserve: int = 4096,
    hz_horizon_s: float = 10.0,
    hz_completion_bonus: float = 1.5,
    hz_prior: str = "swebench9",
    hz_max_batch: int = 64,
    **_,
) -> SchedulingPolicy:
    if name == "size":
        return SizePolicy()
    if name == "density":
        return DensityPolicy(alpha, decode_hat)
    if name == "dual_descent":
        return DualDescentPolicy(alpha, decode_hat, dd_eta0)
    if name == "fidelity":
        return FidelityPolicy(alpha, decode_hat, dd_eta0)
    if name == "hazard_grade":
        return HazardGradePolicy(
            alpha=alpha,
            decode_mean=hz_decode_mean,
            prompt_mean=hz_prompt_mean,
            decode_reserve=hz_decode_reserve,
            horizon_s=hz_horizon_s,
            completion_bonus=hz_completion_bonus,
            prior_name=hz_prior,
        )
    if name == "hazard_grade_v2":
        return HazardGradeV2Policy(
            alpha=alpha,
            decode_mean=hz_decode_mean,
            prompt_mean=hz_prompt_mean,
            decode_reserve=hz_decode_reserve,
            horizon_s=hz_horizon_s,
            completion_bonus=hz_completion_bonus,
            prior_name=hz_prior,
            max_batch=hz_max_batch,
        )
    raise ValueError(f"unknown policy {name!r}; known: {POLICY_NAMES}")
