import json
import os
from pathlib import Path


# Repo root = one level above this file (experiments/)
ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "datasets"
OUTPUT_DIR = DATASETS_DIR / "sft"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUTPUT_DIR / "sft_pairs_v2.jsonl"


def iter_seed_runs():
    """Iterate over seed42/43/44 actions + states."""
    for seed in ["seed42", "seed43", "seed44"]:
        base = DATASETS_DIR / seed
        if not base.exists():
            continue

        actions_dir = base / "actions"
        metrics_dir = base / "metrics"

        if not actions_dir.exists() or not metrics_dir.exists():
            continue

        for action_file in sorted(actions_dir.glob("action_round_*.json")):
            # action_round_X.json -> X
            round_str = action_file.stem.split("_")[-1]
            state_file = metrics_dir / f"state_round_{round_str}.json"
            if not state_file.exists():
                # skip if something is missing
                continue

            source_id = f"{seed}/round_{round_str}"
            yield source_id, action_file, state_file


def iter_old_runs():
    """
    Iterate over old controller traces in:
    datasets/old_runs/run_traces_20251106/<run-id>/
    """
    old_root = DATASETS_DIR / "old_runs" / "run_traces_20251106"
    if not old_root.exists():
        return

    for run_dir in sorted(old_root.iterdir()):
        if not run_dir.is_dir():
            continue

        for action_file in sorted(run_dir.glob("action_round_*.json")):
            round_str = action_file.stem.split("_")[-1]
            state_file = run_dir / f"state_round_{round_str}.json"
            if not state_file.exists():
                continue

            source_id = f"old/{run_dir.name}/round_{round_str}"
            yield source_id, action_file, state_file


def build_sft_pairs():
    """
    Build sft_pairs_v2.jsonl from:
      - datasets/old_runs/run_traces_20251106/*
      - datasets/seed42/43/44
    """

    # If you want to match an older prompt style, change this TEMPLATE.
    TEMPLATE = (
        "You are a federated continual learning controller.\n"
        "Below is the current training STATE as JSON.\n"
        "Your task: choose the next-round controller action.\n"
        "Respond ONLY with a JSON object in the same format as previous actions.\n\n"
        "STATE:\n{state_json}\n"
    )

    all_examples = list(iter_old_runs()) + list(iter_seed_runs())
    if not all_examples:
        print("No examples found. Check DATASETS_DIR paths.")
        return

    count = 0
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as f_out:
        for source_id, action_file, state_file in all_examples:
            with action_file.open("r", encoding="utf-8") as fa:
                action = json.load(fa)
            with state_file.open("r", encoding="utf-8") as fs:
                state = json.load(fs)

            # Optional sanity check: skip non-controller sources if needed
            policy_source = action.get("policy_source", "")
            # If one day you put baseline runs here, you can uncomment this:
            # if "Controller" not in policy_source:
            #     continue

            prompt = TEMPLATE.format(
                state_json=json.dumps(state, indent=2, sort_keys=True)
            )

            # Minified JSON as the target
            response = json.dumps(action, separators=(",", ":"), sort_keys=True)

            record = {
                "source": source_id,
                "prompt": prompt,
                "response": response,
            }
            f_out.write(json.dumps(record) + "\n")
            count += 1

    print(f"Wrote {count} SFT examples to {OUT_PATH}")


if __name__ == "__main__":
    build_sft_pairs()