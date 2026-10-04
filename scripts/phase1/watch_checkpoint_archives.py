#!/usr/bin/env python3
"""Finish the explicitly requested cleanup only after each HF archive verifies."""

import argparse
import json
from pathlib import Path
import time

from scripts.phase1.archive_unused_checkpoints_hf import ALLOWED_STEPS, cleanup_verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    pending = set(ALLOWED_STEPS)
    while pending:
        for step in sorted(pending):
            receipt = args.run / "verl-run/checkpoint_archives" / f"global_step_{step}.json"
            if receipt.is_file():
                cleanup_verified(args.run, step)
                pending.remove(step)
        print(json.dumps({"state": "waiting_for_verified_uploads" if pending else "complete",
                          "pending_steps": sorted(pending)}), flush=True)
        if pending:
            time.sleep(30)


if __name__ == "__main__":
    main()
