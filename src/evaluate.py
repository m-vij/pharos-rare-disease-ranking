"""Evaluate scorers on simulated patients.

Each simulated patient is treated as a NEW patient who has just arrived: every candidate disease is scored against the full
annotation file, and the rank of the true disease gives Recall@K and the reciprocal rank (ties share the rank by expected value).
Add --remove-papers to delete the profile's source papers from the knowledge base first (a much harder test: most diseases
become unrankable, see src/leakage.py).
Results are averaged per disease (so a disease with many patients does not dominate), separately for each profile source, with
95% intervals from resampling diseases.

    python -m src.evaluate --which tuning
    python -m src.evaluate --which heldout --final        # once, when everything is frozen
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from src.scorer import SCORERS, Rarity, make_scorer

KS = (1, 5, 10, 20, 50, 100)


def harmonic(n):
    h = np.zeros(n + 1)
    h[1:] = np.cumsum(1.0 / np.arange(1, n + 1))
    return h


def rank_stats(s, ti, H, ks=KS):
    st = s[ti]
    b, t = int((s > st).sum()), int((s == st).sum())
    out = {f"R@{k}": min(max((k - b) / t, 0.0), 1.0) for k in ks}
    out["MRR"] = float((H[b + t] - H[b]) / t)
    return out


def bootstrap(values, n_boot=2000, seed=0):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate(ann, scorers, cases, n_boot=2000, remove_papers=False):
    """-> {scorer: {source: {'diseases', 'patients', metric: (mean, lo, hi)}}}"""
    idx = {d: i for i, d in enumerate(ann.diseases)}
    H = harmonic(len(ann.diseases))
    views = {}
    rows = {sc.name: defaultdict(lambda: defaultdict(list)) for sc in scorers}
    counts = defaultdict(int)
    for c in cases:
        d = c["disease_id"]
        if d not in idx:
            continue
        key = tuple(sorted(c["exclude_pmids"])) if remove_papers else ()
        if key not in views:
            views[key] = ann.view(key)
        counts[c["source"]] += 1
        for sc in scorers:
            rows[sc.name][(c["source"], d)]["m"].append(rank_stats(sc.score(c["present"], views[key]), idx[d], H))
    out = {}
    for name, groups in rows.items():
        out[name] = {}
        for src in sorted({s for s, _ in groups}):
            per = [np.mean([[m[k] for k in m] for m in g["m"]], axis=0) for (s, _), g in groups.items() if s == src]
            keys = list(next(iter(groups.values()))["m"][0])
            arr = np.array(per)
            out[name][src] = {"diseases": len(per), "patients": counts[src],
                              **{k: bootstrap(arr[:, j], n_boot) for j, k in enumerate(keys)}}
    return out


def print_results(res, shown=("R@1", "R@10", "R@20", "MRR")):
    for src in sorted({s for v in res.values() for s in v}):
        any_row = next(v[src] for v in res.values() if src in v)
        print(f"\nsource: {src}   diseases: {any_row['diseases']}   patients: {any_row['patients']}   (mean over diseases, 95% interval)")
        print(f"{'scorer':<20}" + "".join(f"{k:>24}" for k in shown))
        for name, v in res.items():
            if src in v:
                print(f"{name:<20}" + "".join(f"{v[src][k][0]:>10.3f} [{v[src][k][1]:.3f}, {v[src][k][2]:.3f}]" for k in shown))


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
    ap.add_argument("--scorers", default=",".join(SCORERS))
    ap.add_argument("--sources", default="phenopackets,phenomizer")
    ap.add_argument("--per-profile", type=int, default=0, help="use only the first N patients of each profile (quick runs)")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--remove-papers", action="store_true", help="remove each profile's source papers from the knowledge base (off by default)")
    a = ap.parse_args()
    if a.which == "heldout" and not a.final:
        raise SystemExit("the held-out diseases are for one final run: add --final when every setting is frozen")
    keep = set(json.load(open(a.split))[a.which])
    seen, cases = defaultdict(int), []
    for line in open(a.cases):
        c = json.loads(line)
        if c["disease_id"] in keep and c["source"] in a.sources.split(","):
            seen[c["profile_id"]] += 1
            if not a.per_profile or seen[c["profile_id"]] <= a.per_profile:
                cases.append(c)
    ann = Annotations(a.hpoa, Ontology(a.hpo))
    rarity = Rarity(ann)
    res = evaluate(ann, [make_scorer(n, ann, rarity) for n in a.scorers.split(",")], cases, a.boot, remove_papers=a.remove_papers)
    print(f"split: {a.which}   knowledge base: {ann.version}, papers removed: {a.remove_papers}   patients scored: {len(cases)}")
    print_results(res)


if __name__ == "__main__":
    main()
