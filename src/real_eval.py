"""Real patients with LEAVE-SOURCE-OUT scoring (a secondary analysis, separate from the simulated study).

Every real patient is scored against the annotation file WITHOUT the paper that reported them (all annotations citing it are
removed). Patients are grouped by what that removal does to their own disease:
  real_none_removed     none of the disease's annotations came from that paper
  real_partly_removed   some did
  real_all_removed      all did: the disease has nothing left to match, so no scorer can rank it
and `real_all` pools them. Results are averaged per disease, with 95% intervals from resampling diseases.
Diseases in the TUNING split are left out by default: prompts and settings were developed on their simulated patients.
Nothing here changes any frozen file or setting.

    python -m src.real_eval
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from src.evaluate import bootstrap, harmonic, print_results, rank_stats
from src.scorer import SCORERS, Rarity, make_scorer


def stratum(frac):
    return "real_all_removed" if frac >= 1 - 1e-9 else "real_none_removed" if frac <= 1e-9 else "real_partly_removed"


def build_cases(real_cases, ann, exclude_diseases=(), per_disease=0, skip_all_removed=False):
    """Real patients as case dicts in the simulator's format; `source` carries the stratum, `exclude_pmids` the patient's own paper(s)."""
    ok, ex = set(ann.diseases), set(exclude_diseases)
    views, out, seen = {}, [], defaultdict(int)
    for r in sorted(real_cases, key=lambda r: r["case_id"]):
        d = r["disease_id"]
        if d not in ok or d in ex:
            continue
        key = tuple(sorted(r["pmids"]))
        if key not in views:
            views[key] = ann.view(key)
        s = stratum(views[key].removed_fraction(d))
        if skip_all_removed and s == "real_all_removed":
            continue
        if per_disease and seen[d] >= per_disease:
            continue
        seen[d] += 1
        out.append({"case_id": r["case_id"], "profile_id": r["case_id"], "disease_id": d, "disease_label": r["disease_label"], "source": s,
                    "sex": r.get("sex"), "present": r["present"], "exclude_pmids": list(r["pmids"])})
    return out


def evaluate_real(ann, scorers, cases, n_boot=2000):
    """-> {scorer: {group: {'diseases', 'patients', metric: (mean, lo, hi)}}} for each stratum and for 'real_all' (one scoring pass)."""
    idx = {d: i for i, d in enumerate(ann.diseases)}
    H = harmonic(len(ann.diseases))
    views = {}
    per = {sc.name: defaultdict(list) for sc in scorers}
    counts = defaultdict(int)
    for c in cases:
        key = tuple(sorted(c["exclude_pmids"]))
        if key not in views:
            views[key] = ann.view(key)
        for g in (c["source"], "real_all"):
            counts[g] += 1
        for sc in scorers:
            m = rank_stats(sc.score(c["present"], views[key]), idx[c["disease_id"]], H)
            per[sc.name][(c["source"], c["disease_id"])].append(m)
            per[sc.name][("real_all", c["disease_id"])].append(m)
    out = {}
    for name, groups in per.items():
        out[name] = {}
        for g in sorted({k[0] for k in groups}):
            rows = [np.mean([[m[k] for k in m] for m in ms], axis=0) for (gg, _), ms in groups.items() if gg == g]
            keys = list(next(iter(groups.values()))[0])
            arr = np.array(rows)
            out[name][g] = {"diseases": len(rows), "patients": counts[g], **{k: bootstrap(arr[:, j], n_boot) for j, k in enumerate(keys)}}
    return out


def load_real(path):
    return [json.loads(l) for l in open(path)]


def main():
    from src.annotations import Annotations
    from src.ontology import Ontology
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="cache/cases.jsonl")
    ap.add_argument("--hpo", default="hp.obo")
    ap.add_argument("--hpoa", default="phenotype.hpoa")
    ap.add_argument("--split", default="config/split.json")
    ap.add_argument("--include-tuning", action="store_true", help="also score diseases of the tuning split")
    ap.add_argument("--scorers", default=",".join(SCORERS))
    ap.add_argument("--per-disease", type=int, default=0, help="at most N patients per disease (0 = all)")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    exclude = set() if a.include_tuning else set(json.load(open(a.split))["tuning"])
    ann = Annotations(a.hpoa, Ontology(a.hpo))
    cases = build_cases(load_real(a.cases), ann, exclude, a.per_disease)
    n = defaultdict(int)
    for c in cases:
        n[c["source"]] += 1
    print(f"real patients scored: {len(cases)} from {len({c['disease_id'] for c in cases})} diseases   knowledge base: {ann.version}   "
          f"tuning diseases excluded: {not a.include_tuning}")
    print("strata: " + ", ".join(f"{k} {v}" for k, v in sorted(n.items())))
    rarity = Rarity(ann)
    print_results(evaluate_real(ann, [make_scorer(s, ann, rarity) for s in a.scorers.split(",")], cases, a.boot))


if __name__ == "__main__":
    main()
