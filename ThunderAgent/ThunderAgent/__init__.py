"""
ThunderAgent - VLLM proxy with program state tracking.

Module structure:
- backend: Backend state management
- program: Program state management
- scheduler: Request routing and proxying
"""

from .config import Config, get_config, set_config
from .backend import BackendState
from .program import ProgramState, ProgramStatus
from .scheduler import MultiBackendRouter

# NOTE: do NOT `from .app import ...` here. app.py builds the router singleton at
# import time (router = _create_router(), reading get_config()). Importing app
# from this package __init__ makes that happen on the very first `import
# ThunderAgent...` -- which in __main__ is BEFORE set_config() runs -- so the
# router would be built with the DEFAULT config (policy=size, scheduler=base,
# interval=5.0), silently ignoring all CLI flags. Keep app out of __init__ so the
# router is only built by __main__'s `from .app import app` AFTER set_config().
# (Import get_program_id / register_routes directly from ThunderAgent.app if needed.)

__all__ = [
    "Config",
    "get_config",
    "set_config",
    "BackendState",
    "ProgramState",
    "ProgramStatus",
    "MultiBackendRouter",
]

__version__ = "0.2.0"
