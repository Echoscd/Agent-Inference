"""DensityPolicy: value-density ordering v = 1 / (tau * footprint).

Holds the two scheduling constants (alpha, decode_hat) and the service-time /
density math that used to live on the Router (_service_time, _program_density).
"""
from __future__ import annotations

from .base import SchedulingPolicy
from ..program import Program


class DensityPolicy(SchedulingPolicy):
    name = "density"

    def __init__(self, alpha: float, decode_hat: float):
        self.alpha = alpha
        self.decode_hat = decode_hat

    def service_time(self, state: Program) -> float:
        """tau = alpha * footprint + decode (cold model; the whole footprint pays
        prefill cost). decode = known_decode (X-Decode-Len) when available, else the
        offline decode_hat constant. FidelityPolicy overrides to discount the warm
        prefix."""
        q = max(1.0, float(state.total_tokens))
        decode = float(state.known_decode) if state.known_decode > 0 else self.decode_hat
        return self.alpha * q + decode

    def density(self, state: Program) -> float:
        """value-density v = 1 / (tau * footprint). Higher = more value per
        resource-second -> admit/keep first."""
        q = max(1.0, float(state.total_tokens))
        return 1.0 / (self.service_time(state) * q)

    def sort_key(self, state: Program):
        # larger density = higher keep-priority; the key IS the value.
        return (self.density(state),)
