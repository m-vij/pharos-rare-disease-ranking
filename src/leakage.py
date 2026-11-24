"""How much of each profile's disease knowledge does the paper removal take away, and how does recall depend on it?

A cohort profile stands for everything its papers reported, so ALL of those papers are removed from the knowledge base when its
patients are scored. For a disease whose annotations rest mostly on those same papers, little independent knowledge is left, and no
scorer can rank it. This groups the profiles by the share of their disease's annotations that the removal deleted and gives recall
for each group.

    python -m src.leakage --which tuning
"""
import argparse
import json
from collections import defaultdict

from src.evaluate import evaluate
from src.scorer import Rarity, make_scorer

BUCKETS = ["100% removed", "75-99% removed", "50-74% removed", "25-49% removed", "under 25% removed"]


def bucket_of(frac):
    if frac >= 1.0 - 1e-9:
        return BUCKETS[0]
    return BUCKETS[1] if frac >= 0.75 else BUCKETS[2] if frac >= 0.5 else BUCKETS[3] if frac >= 0.25 else BUCKETS[4]


def removal_by_profile(ann, cases):
    out, views = {}, {}
    for c in cases:
        if c["profile_id"] not in out:
            key = tuple(sorted(c["exclude_pmids"]))
            if key not in views:
                views[key] = ann.view(key)
            out[c["profile_id"]] = views[key].removed_fraction(c["disease_id"])
    return out


def by_bucket(ann, scorers, cases, n_boot=1000):
    frac = removal_by_profile(ann, cases)
    groups = defaultdict(list)
    for c in cases:
        groups[bucket_of(frac[c["profile_id"]])].append(c)
    out = {}
    for b in BUCKETS:
        if b in groups:
            out[b] = {"profiles": len({c["profile_id"] for c in groups[b]}), "results": evaluate(ann, scorers, groups[b], n_boot, remove_papers=True)}
    return out


def main():
    from src.annotations import Annotations
    from src.ontology import Ontology
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="cache/sim_cases.jsonl")
    ap.add_argument("--hpo", default="hp.obo")
    ap.add_argument("--hpoa", default="phenotype.hpoa")
    ap.add_argument("--split", default="config/split.json")
    ap.add_argument("--which", default="tuning", choices=["tuning", "heldout"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--scorers", default="exact_overlap,resnik_best_match")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    if a.which == "heldout" and not a.final:
        raise SystemExit("the held-out diseases are for one final run: add --final when every setting is frozen")
    keep = set(json.load(open(a.split))[a.which])
    cases = [c for c in map(json.loads, open(a.cases)) if c["disease_id"] in keep and c["source"] == "phenopackets"]
    ann = Annotations(a.hpoa, Ontology(a.hpo))
    rarity = Rarity(ann)
    scorers = [make_scorer(n, ann, rarity) for n in a.scorers.split(",")]
    res = by_bucket(ann, scorers, cases, a.boot)
    print(f"cohort profiles, split {a.which}: share of the disease's annotations deleted by removing the profile's papers")
    print(f"{'removed':<20}{'profiles':>9}" + "".join(f"{n[:18] + ' R@10':>26}{n[:18] + ' MRR':>26}" for n in a.scorers.split(",")))
    for b, v in res.items():
        cells = ""
        for n in a.scorers.split(","):
            r = v["results"][n]["phenopackets"]
            cells += f"{r['R@10'][0]:>10.3f} [{r['R@10'][1]:.3f}, {r['R@10'][2]:.3f}]  {r['MRR'][0]:>8.3f} [{r['MRR'][1]:.3f}, {r['MRR'][2]:.3f}]"
        print(f"{b:<20}{v['profiles']:>9}" + cells)


if __name__ == "__main__":
    main()
