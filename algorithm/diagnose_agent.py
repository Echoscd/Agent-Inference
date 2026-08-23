"""Run the edit-agent on one instance, save the FULL transcript (every message +
parsed action per turn) to result/05_agent_behavior/, to diagnose agent behaviour
(e.g. why a run ends after few turns). English-only output."""
import os, json, sys
import swebench_edit_agent as A
import paths

def main():
    OUTDIR = paths.result("05_agent_behavior")
    os.makedirs(OUTDIR, exist_ok=True)

    iid = sys.argv[1] if len(sys.argv) > 1 else "psf__requests-1766"
    A.MODEL = "qwen3-32b"

    transcript = []   # list of {turn, assistant, action_kind, action_payload_keys, observation}
    orig_parse = A.parse_action
    _turn = [0]
    def parse_capture(text):
        k, p = orig_parse(text)
        _turn[0] += 1
        transcript.append({"turn": _turn[0], "assistant": text,
                           "action_kind": k, "payload_keys": list(p.keys())})
        return k, p
    A.parse_action = parse_capture

    inst = next(json.loads(l) for l in open(paths.SWEBENCH_DATA)
                if json.loads(l)["instance_id"] == iid)
    r = A.run_agent(inst, verbose=False, max_turns=8)

    # attach the observations the agent saw (reconstruct from result is not stored; we logged actions)
    out = {
        "instance_id": iid, "repo": inst["repo"], "version": inst["version"],
        "turns": r.turns, "submitted": r.submitted, "status": r.status,
        "resolved": r.resolved, "gen_tokens": r.total_gen_tokens,
        "prompt_tokens": r.total_prompt_tokens, "patch_len": len(r.patch),
        "error": r.error,
        "turn_actions": [{"turn": t["turn"], "action_kind": t["action_kind"],
                          "payload_keys": t["payload_keys"],
                          "assistant_chars": len(t["assistant"])} for t in transcript],
    }
    json.dump(out, open(os.path.join(OUTDIR, f"{iid}_summary.json"), "w"), indent=2)
    with open(os.path.join(OUTDIR, f"{iid}_transcript.txt"), "w") as f:
        f.write(f"instance: {iid} ({inst['repo']} {inst['version']})\n")
        f.write(f"result: turns={r.turns} submitted={r.submitted} status={r.status} patch_len={len(r.patch)}\n")
        if r.error: f.write(f"error: {r.error}\n")
        f.write("=" * 80 + "\n")
        for t in transcript:
            f.write(f"\n----- TURN {t['turn']}  -> action={t['action_kind']} {t['payload_keys']} -----\n")
            f.write(t["assistant"] + "\n")

    print(f"saved transcript to {OUTDIR}/{iid}_transcript.txt")
    print(f"turns={r.turns} submitted={r.submitted} status={r.status} patch_len={len(r.patch)}")
    print("per-turn actions:", [(t["turn"], t["action_kind"]) for t in transcript])


if __name__ == "__main__":
    main()
