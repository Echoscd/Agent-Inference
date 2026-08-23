"""FidelityPolicy: DualDescent + the three fidelity gaps vs the offline simulator.
Replaces the former FidelityRouter subclass -- these are policy decisions, not
Router orchestration.

  Gap 1 (peak reservation, var 1): reserve the upcoming decode at admission.
  Gap 6 (warm/cold service time, vars 2/5): discount the prefix-cached portion of tau.
  Gap 3 (proactive eviction, var 4): pause below-price ACTING programs each epoch.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable

from .dual_descent import DualDescentPolicy
from ..program import Program, ProgramStatus
from ..backend import BackendState

logger = logging.getLogger(__name__)


class FidelityPolicy(DualDescentPolicy):
    name = "fidelity"

    # Gap 1 (var 1: admission amount = peak reservation)
    def peak_pad(self, state: Program) -> int:
        return int(state.known_decode) if state.known_decode > 0 else int(self.decode_hat)

    # Gap 6 (vars 2/5: warm/cold service time). Overriding service_time makes
    # density and sort_key warm-aware automatically. Only the NON-cached portion of
    # the context pays prefill cost; warm fraction = cached/prompt tokens last turn.
    def service_time(self, state: Program) -> float:
        q = max(1.0, float(state.total_tokens))
        decode = float(state.known_decode) if state.known_decode > 0 else self.decode_hat
        warm = (
            min(1.0, max(0.0, state.last_cached_tokens / state.last_prompt_tokens))
            if state.last_prompt_tokens > 0
            else 0.0
        )
        return self.alpha * q * (1.0 - warm) + decode

    # Gap 3 (var 4: eviction timing = proactive, price-triggered). Only ACTING
    # programs (between turns, safe to pause) -- never REASONING (mid-decode),
    # preserving the simulator's non-preemptive, request-boundary semantics.
    # Starvation is bounded by the Router's force-resume timeout.
    def proactive_evictions(
        self, backends: Dict[str, BackendState], programs: Dict[str, Program]
    ) -> Iterable[str]:
        victims = []
        for url, b in backends.items():
            if not b.healthy:
                continue
            for pid, st in programs.items():
                if (
                    st.backend_url == url
                    and st.status == ProgramStatus.ACTING
                    and self.density(st) - self.lam <= 0.0
                ):
                    victims.append(pid)
        if victims:
            logger.info(
                f"FidelityPolicy: proactively evicting {len(victims)} below-price "
                f"ACTING program(s) (lambda={self.lam:.3e})"
            )
        return victims
