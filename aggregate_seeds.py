import argparse
import json
from pathlib import Path
import numpy as np

SEEDS = (111, 222, 333, 444, 555)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs=5, required=True, help="Five results.json files in seed order 111..555")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in args.files]
    if tuple(r.get("seed") for r in rows) != SEEDS:
        raise ValueError(f"Files must be supplied IN ORDER for actual seeds {SEEDS}")
    if len({r["dataset"] for r in rows}) != 1:
        raise ValueError("Seed files must evaluate the same dataset")
    if any(r["partial_debug_run"] or r["feature_space"] != "pixel" or not r.get("metrics") for r in rows):
        raise ValueError("Cannot publish partial, non-paper, or absent results")
    for field in ("paper_config", "prompt_setting", "metadata_count", "processed"):
        if any(r[field] != rows[0][field] for r in rows):
            raise ValueError(f"Mixed settings across seeds: {field}")
    stats = {k: {"mean": float(np.mean([r["metrics"][k] for r in rows])),
                 "std_ddof1": float(np.std([r["metrics"][k] for r in rows], ddof=1))}
             for k in ("AuROC", "AP", "FPR95")}
    data = {"dataset": rows[0]["dataset"], "seeds": SEEDS, "statistics": stats,
            "note": "Actual rerun measurements only; seed-wise sample SD ddof=1"}
    Path(args.output).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
