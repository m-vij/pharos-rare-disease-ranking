"""Fixed split of the diseases into TUNING (used to choose settings) and HELD-OUT (touched once, at the end).

The split is by disease (a disease's patients, from both sources, always fall on the same side), seeded, and written once to
config/split.json. It refuses to overwrite an existing file, because moving a disease across the line after seeing results
would make the held-out numbers meaningless.

    python -m src.split
"""
import argparse
import json
import random
from pathlib import Path


def make_split(disease_ids, seed=20260415, tuning_share=0.3):
    ids = sorted(set(disease_ids))
    random.Random(seed).shuffle(ids)
    k = round(len(ids) * tuning_share)
    return {"seed": seed, "tuning": sorted(ids[:k]), "heldout": sorted(ids[k:])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="cache/profiles.json")
    ap.add_argument("--out", default="config/split.json")
    ap.add_argument("--seed", type=int, default=20260415)
    ap.add_argument("--tuning-share", type=float, default=0.3)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if Path(a.out).exists() and not a.force:
        raise SystemExit(f"{a.out} already exists and must not change; delete it by hand only if you really intend to re-split")
    split = make_split([p["disease_id"] for p in json.load(open(a.profiles))], a.seed, a.tuning_share)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(split, open(a.out, "w"), indent=1)
    print(f"{len(split['tuning'])} tuning diseases, {len(split['heldout'])} held-out diseases -> {a.out}")


if __name__ == "__main__":
    main()
