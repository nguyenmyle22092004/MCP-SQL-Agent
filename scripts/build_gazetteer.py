from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from decision_layer.local.gazetteer import build_gazetteer_from_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the cached house-price gazetteer using a read-only PostgreSQL role.")
    parser.add_argument("--output", default="data/gazetteer.json")
    args = parser.parse_args()
    gazetteer = build_gazetteer_from_database(args.output)
    print(f"Wrote {args.output}: {len(gazetteer['cities'])} cities, {len(gazetteer['states'])} states, {len(gazetteer['street_names'])} street names.")


if __name__ == "__main__":
    main()
