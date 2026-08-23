# 05_agent_behavior

Single-instance transcripts used to diagnose why runs ended after few turns.

**The original transcripts were lost on 2026-08-23.** `diagnose_agent.py` had no
`if __name__ == "__main__"` guard, so merely importing the module ran it; an
import during a housekeeping check executed it against a backend that was not
up, and it overwrote the saved transcript with a 0-turn connection error. The
missing guard has since been fixed. Regenerate with:

    python3 algorithm/diagnose_agent.py psf__requests-1766   # needs a live backend

Note the agent has changed since June, so a regenerated transcript will not match
the original.

## What the original run established

Two harness bugs, both fixed at the time:

1. `EDIT` and `SUBMIT` in one message dropped the edit — the action parser gave
   `SUBMIT` priority.
2. `SEARCH/REPLACE` matched exactly, so correct fixes were rejected over
   indentation. Matching is whitespace-tolerant now.

Model output was complete (`finish_reason=stop`) and often correct. The low
resolve rate at that point was the scaffolding, not the inference.
