"""ThunderAgent entry point for `python -m ThunderAgent`."""
import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ThunderAgent - Program State Tracking Proxy for vLLM",
        prog="python -m ThunderAgent",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8300, help="Port to bind to")
    parser.add_argument("--log-level", default="info", help="Log level")
    parser.add_argument("--backends", default="http://localhost:8000", 
                        help="Comma-separated list of vLLM backend URLs")
    parser.add_argument("--router", default="tr", choices=["default", "tr"],
                        help="Router mode: 'default' (pure proxy) or 'tr' (capacity scheduling)")
    parser.add_argument("--backend-type", default="vllm", choices=["vllm", "sglang", "skyrl"],
                        help="Backend type: 'vllm', 'sglang', or 'skyrl'")
    parser.add_argument("--profile", action="store_true", 
                        help="Enable profiling (track prefill/decode/tool_call times)")
    parser.add_argument("--profile-dir", default="/tmp/thunderagent_profiles", 
                        help="Directory for profile CSV output")
    parser.add_argument("--metrics", action="store_true",
                        help="Enable vLLM metrics monitoring")
    parser.add_argument("--metrics-interval", type=float, default=5.0,
                        help="Interval in seconds between metrics fetches (default: 5.0)")
    parser.add_argument("--scheduler-interval", type=float, default=5.0,
                        help="Interval in seconds between scheduler checks (default: 5.0)")
    parser.add_argument("--acting-token-weight", type=float, default=1.0,
                        help="Weight for acting tokens in capacity calculation (default: 1.0)")
    parser.add_argument("--use-acting-token-decay", action="store_true",
                        help="Use 2^(-t) decay for acting tokens in resume capacity calculation")
    parser.add_argument("--policy", default="size",
                        choices=["size", "density", "dual_descent", "fidelity", "hazard_grade",
                                 "sim0823", "bdp"],
                        help="Scheduling policy: 'size' (token-size ordering), "
                             "'density' (admit/keep highest 1/(tau*footprint), evict lowest), "
                             "'dual_descent' (density minus a learned KV price; admit only density>lambda), "
                             "'fidelity' (dual_descent + peak reservation + proactive eviction + warm/cold tau), or "
                             "'hazard_grade' (prediction-light admission grade + hazard cache 0/1 knapsack)")
    parser.add_argument("--alpha", type=float, default=0.03,
                        help="density/dual_descent/hazard_grade: prefill-token cost relative to one decode token (offline-fixed from historical p/d)")
    parser.add_argument("--decode-hat", type=float, default=1000.0,
                        help="density/dual_descent: KNOWN decode tokens/turn used in tau (assumed given; offline constant)")
    parser.add_argument("--dd-eta0", type=float, default=0.10,
                        help="dual_descent: dual-price step-size scale eta0 (default: 0.10)")
    # hazard_grade knobs
    parser.add_argument("--hz-decode-mean", type=float, default=1650.0,
                        help="hazard_grade: mean decode tokens/turn used in the run_score work terms")
    parser.add_argument("--hz-prompt-mean", type=float, default=1800.0,
                        help="hazard_grade: mean incremental prompt tokens/turn")
    parser.add_argument("--hz-decode-reserve", type=int, default=4096,
                        help="hazard_grade: Q~95 decode reservation for peak_pad (NOT the oracle known_decode)")
    parser.add_argument("--hz-horizon-s", type=float, default=10.0,
                        help="hazard_grade: tool return-probability lookahead horizon in seconds")
    parser.add_argument("--hz-completion-bonus", type=float, default=1.5,
                        help="hazard_grade: weight on posterior terminal probability in cache value")
    parser.add_argument("--bdp-context-limit", type=int, default=32768,
                        help="bdp: context guard where the server rejects the next prompt")
    parser.add_argument("--hz-max-batch", type=int, default=64,
                        help="sim0823: per-tick admission cap (the simulator's max_batch)")
    parser.add_argument("--hz-prior", default="swebench9",
                        choices=["short", "swebench9", "quick10", "long", "coder16"],
                        help="hazard_grade: round-count prior name (default: swebench9; coder16 for non-CoT coder)")
    args = parser.parse_args()

    # Set config BEFORE importing app
    from .config import Config, set_config
    
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    config = Config(
        backends=backends,
        router_mode=args.router,
        backend_type=args.backend_type,
        profile_enabled=args.profile,
        profile_dir=args.profile_dir,
        metrics_enabled=args.metrics,
        metrics_interval=args.metrics_interval,
        scheduler_interval=args.scheduler_interval,
        acting_token_weight=args.acting_token_weight,
        use_acting_token_decay=args.use_acting_token_decay,
        policy=args.policy,
        alpha=args.alpha,
        decode_hat=args.decode_hat,
        dd_eta0=args.dd_eta0,
        hz_decode_mean=args.hz_decode_mean,
        hz_prompt_mean=args.hz_prompt_mean,
        hz_decode_reserve=args.hz_decode_reserve,
        hz_horizon_s=args.hz_horizon_s,
        hz_completion_bonus=args.hz_completion_bonus,
        hz_prior=args.hz_prior,
        hz_max_batch=args.hz_max_batch,
        bdp_context_limit=args.bdp_context_limit,
    )
    set_config(config)
    
    print(f"🚀 Router mode: {args.router}")
    if args.profile:
        print(f"📊 Profiling enabled - CSV output: {args.profile_dir}/step_profiles.csv")
    
    if args.metrics:
        print(f"📈 Metrics monitoring enabled - interval: {args.metrics_interval}s")
    
    if args.router == "tr":
        print(f"⏱️  Scheduler interval: {args.scheduler_interval}s")
        print(f"⚖️  Acting token weight: {args.acting_token_weight}")
        if args.use_acting_token_decay:
            print(f"📉 Acting token decay: enabled (2^-t)")
        print(f"🧮 Scheduling policy: {args.policy}")
        if args.policy in ("density", "dual_descent", "fidelity"):
            print(f"   density tau = {args.alpha}*footprint + {args.decode_hat} (decode assumed known)")
        if args.policy in ("dual_descent", "fidelity"):
            print(f"   dual price: learned KV price lambda, eta0={args.dd_eta0} (admit only density>lambda)")
        if args.policy == "fidelity":
            print(f"   fidelity: peak reservation + proactive price eviction + warm/cold tau (Gaps 1/3/6)")
        if args.policy in ("hazard_grade", "sim0823", "bdp"):
            print(f"   hazard_grade: run_score=1/(current_work*remaining_work*blocks), no price gate;")
            print(f"     peak_pad={args.hz_decode_reserve} (Q95 reservation, NOT known_decode);")
            print(f"     cache knapsack: value=cold*P(tool<=+{args.hz_horizon_s}s)*(1+{args.hz_completion_bonus}*term_prob), prior={args.hz_prior}")

    # Import uvicorn here to avoid import errors if not installed
    try:
        import uvicorn
    except ImportError:
        print("Error: uvicorn is required. Install with: pip install uvicorn", file=sys.stderr)
        return 1

    # Import app after config is set
    from .app import app
    # FIX: app.py creates `router` at import time (triggered by package __init__),
    # which runs BEFORE set_config() above -> router got profile_enabled=False.
    # Re-sync the live router with the now-correct config so profiling actually runs.
    from . import app as _appmod
    _appmod.router.profile_enabled = config.profile_enabled
    print(f"🔧 router.profile_enabled synced -> {_appmod.router.profile_enabled}")

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
