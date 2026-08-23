"""ThunderAgent configuration."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    """ThunderAgent configuration (set via command line args)."""
    # Backend configuration
    backends: List[str] = field(default_factory=lambda: ["http://localhost:8000"])
    
    # Router mode: "default" (pure proxy) or "tr" (capacity scheduling)
    router_mode: str = "tr"

    # Backend type: "vllm", "sglang", or "skyrl"
    backend_type: str = "vllm"
    
    # Profile configuration
    profile_enabled: bool = False
    profile_dir: str = "/tmp/thunderagent_profiles"
    
    # Metrics monitoring configuration
    metrics_enabled: bool = False
    metrics_interval: float = 5.0  # seconds between metrics fetch
    
    # Scheduler configuration
    scheduler_interval: float = 5.0  # seconds between scheduler checks
    acting_token_weight: float = 1.0  # weight for acting tokens in capacity calculation
    use_acting_token_decay: bool = False  # use 2^(-t) decay for acting tokens in resume logic

    # Scheduling policy (a SchedulingPolicy in ThunderAgent/scheduling/):
    #   "size"         -> priority groups, token-size ascending
    #   "density"      -> value-density v = 1/(tau*footprint), admit/keep highest first
    #   "dual_descent" -> density minus a learned KV price lambda; admit only density>lambda
    #   "fidelity"     -> dual_descent + peak reservation + proactive eviction + warm/cold tau
    #   "hazard_grade" -> prediction-light admission grade + hazard cache 0/1 knapsack
    policy: str = "size"
    alpha: float = 0.03         # prefill-token cost relative to one decode token (offline, from historical p/d)
    decode_hat: float = 1000.0  # KNOWN decode tokens/turn (assumed given for now; offline constant)
    dd_eta0: float = 0.10       # dual_descent: dual-price step-size scale (eta0)

    # hazard_grade knobs (see scheduling/hazard_grade.py; class-level defaults,
    # replace with fits from real ThunderAgent traces before trusting effect sizes)
    hz_decode_mean: float = 1650.0     # mean decode tokens/turn (for run_score work terms)
    hz_prompt_mean: float = 1800.0     # mean incremental prompt tokens/turn
    hz_decode_reserve: int = 4096      # Q~95 decode reservation for peak_pad (NOT the oracle)
    hz_horizon_s: float = 10.0         # tool return-probability lookahead horizon (s)
    hz_completion_bonus: float = 1.5   # weight on posterior terminal probability in cache value
    hz_prior: str = "swebench9"        # round-count prior name (tool_hazard.ROUND_PRIORS)


# Global config instance (set by __main__.py before app starts)
_config: Config = Config()


def get_config() -> Config:
    """Get the global config instance."""
    return _config


def set_config(config: Config) -> None:
    """Set the global config instance."""
    global _config
    _config = config
