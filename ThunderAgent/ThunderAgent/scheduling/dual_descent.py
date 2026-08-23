"""DualDescentPolicy: density minus a learned KV price lambda.

value = density - lambda (same ordering as density, lambda uniform). The price
gates admission (admit only density > lambda, leaving headroom) and is updated each
epoch by a dual-subgradient step on demand/capacity. Port of DualDescentCurrentPolicy
in the offline simulator.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, Iterable

from .density import DensityPolicy
from ..program import Program
from ..backend import BackendState

logger = logging.getLogger(__name__)


class DualDescentPolicy(DensityPolicy):
    name = "dual_descent"

    # Cap lambda growth to at most +20% per epoch (cold-start spike guard). Decreases
    # stay uncapped so the price can still fall fast once KV pressure clears.
    MAX_LAMBDA_RISE = 1.20

    def __init__(self, alpha: float, decode_hat: float, eta0: float):
        super().__init__(alpha, decode_hat)
        self.eta0 = eta0
        self.lam = 0.0          # current dual price lambda_t
        self.epoch = 0          # epoch counter (diminishing step)
        self.scale = None       # set once = median density

    # Variables 2 & 5: reduced value-density (same direction as density)
    def sort_key(self, state: Program):
        return (self.density(state) - self.lam,)

    # Variable 3: price gate -> leave KV headroom under pressure
    def admits(self, state: Program) -> bool:
        return self.density(state) - self.lam > 0.0

    # Per-epoch: dual-(sub)gradient update of lambda
    #   lambda_{t+1} = max(0, lambda_t + eta_t * (desired/free - 1))
    #   eta_t = eta0 * scale / sqrt(epoch+1),  scale = median density (fixed once)
    #   desired = footprint of WAITING programs that clear the price (density > lambda)
    #   free    = free KV capacity across healthy backends
    def on_epoch(self, backends: Dict[str, BackendState], waiting: Dict[str, Program]) -> None:
        free = sum(
            b.remaining_capacity()
            for b in backends.values()
            if b.healthy and b.cache_config
        )
        # Guard (mirrors the simulator's `if free <= 0 or not waiting: return`):
        # do NOT update lambda when KV is saturated (free <= 0) or nothing is
        # waiting. Without this, at free==0 the `desired / max(1, free)` term
        # degenerates into raw token count and lambda overshoots by orders of
        # magnitude, permanently jamming the admission gate.
        if free <= 0 or not waiting:
            return
        densities = []
        desired = 0.0
        for _pid, st in waiting.items():
            d = self.density(st)
            densities.append(d)
            if d - self.lam > 0.0:
                desired += float(st.total_tokens)
        if self.scale is None and densities:
            s = sorted(densities)
            med = s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])
            self.scale = max(med, 1e-12)
        scale = self.scale if self.scale is not None else 1e-9
        step = self.eta0 * scale / math.sqrt(self.epoch + 1.0)
        proposed = max(0.0, self.lam + step * (desired / max(1.0, float(free)) - 1.0))
        # Per-epoch rise cap: lambda climbs at most +20% per epoch, taming the
        # cold-start spike where an all-at-once arrival burst drives free->~0 and
        # lambda overshoots in a few epochs, freezing admission. The first rise off
        # zero is exempt so the price can bootstrap (0 * 1.20 would pin it at 0).
        if self.lam > 0.0 and proposed > self.lam:
            proposed = min(proposed, self.lam * self.MAX_LAMBDA_RISE)
        self.lam = proposed
        self.epoch += 1
        logger.info(
            f"dual_descent: epoch={self.epoch} lambda={self.lam:.6e} "
            f"desired={desired:.0f} free={free} waiting={len(waiting)}"
        )
