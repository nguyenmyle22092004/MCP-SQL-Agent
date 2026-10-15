from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_layer.local.evaluation import load_jsonl
from decision_layer.local.model import save_artifact, train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the pure-Python local decision model from labeled train/test JSONL.")
    parser.add_argument("--dataset", required=True, help="Labeled JSONL with an explicit split field.")
    parser.add_argument("--gazetteer", default="data/gazetteer.json")
    parser.add_argument("--output", default="decision_layer/artifacts/local_model.json")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--calibration", choices=("platt", "isotonic"), default="platt")
    args = parser.parse_args()
    import json

    gazetteer = json.loads(Path(args.gazetteer).read_text(encoding="utf-8"))
    records = load_jsonl(args.dataset)
    artifact = train_model(records, gazetteer, seed=args.seed, calibration=args.calibration)
    save_artifact(artifact, args.output)
    print(f"Trained on {artifact['training_rows']} train rows; artifact: {args.output}")
    print(f"Training data SHA-256: {artifact['training_data_sha256']}")


if __name__ == "__main__":
    main()
