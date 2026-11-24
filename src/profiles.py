"""Disease profiles: for each disease, how often each finding occurs. Simulated patients are drawn from these.

Two sources, kept as separate profiles (a disease may have both):
  phenopackets  the share of the disease's real patients (cohort) who have each term; needs >= min_cases patients.
                Terms are used as recorded (no propagation), so a simulated patient is about as long as a real one.
  phenomizer    the 44 syndromes of the Phenomizer supplement, with their printed frequencies. HPO IDs are repaired
                automatically (src/resolve_ids.py); features that cannot be resolved are dropped and logged.

A profile only exists for diseases that are candidates in the annotation file (otherwise they could never be ranked).

    python -m src.profiles
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.resolve_ids import Resolver


def cohort_profiles(cases, min_cases=10):
    by = defaultdict(list)
    for c in cases:
        by[c["disease_id"]].append(c)
    out = []
    for d, cs in sorted(by.items()):
        n = len(cs)
        if n < min_cases:
            continue
        counts = Counter(t for c in cs for t in set(c["present"]))
        feats = [{"terms": [t], "op": "single", "freq": counts[t] / n, "sex": None}
                 for t in sorted(counts, key=lambda t: (-counts[t], t))]
        out.append({"id": f"{d}|phenopackets", "disease_id": d, "name": cs[0]["disease_label"], "source": "phenopackets",
                    "n_cases": n, "pmids": sorted({p for c in cs for p in c["pmids"]}), "features": feats})
    return out


def phenomizer_profiles(tables, onto, ann_diseases):
    """-> (profiles, log rows, skipped syndromes)."""
    res = Resolver(onto)
    profiles, rows, skipped = [], [], []
    for t in tables:
        d = t["omim"]
        if d is None:
            skipped.append((t["name"], "no OMIM number printed"))
            continue
        if d not in ann_diseases:
            skipped.append((t["name"], f"{d} is not a candidate in the annotation file"))
            continue
        feats = []
        for f in t["features"]:
            ids, log = res.resolve_feature(f["name"], f["ids"])
            for e in log:
                rows.append({"table": t["table"], "feature": f["name"], **e})
            if not ids or f["freq"] is None:
                rows.append({"table": t["table"], "feature": f["name"], "printed": "", "result": None, "method": "feature dropped: " + ("no usable id" if not ids else "no frequency"), "similarity": None})
                continue
            feats.append({"terms": ids, "op": f["op"] if len(ids) > 1 else "single", "freq": f["freq"], "sex": f["sex"]})
        profiles.append({"id": f"{d}|phenomizer", "disease_id": d, "name": t["name"], "source": "phenomizer",
                         "n_cases": None, "pmids": [], "features": feats})
    return profiles, rows, skipped


def main():
    from src.annotations import Annotations
    from src.ontology import Ontology
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="cache/cases.jsonl")
    ap.add_argument("--tables", default="data/phenomizer_tables.json")
    ap.add_argument("--hpo", default="hp.obo")
    ap.add_argument("--hpoa", default="phenotype.hpoa")
    ap.add_argument("--min-cases", type=int, default=10)
    ap.add_argument("--out", default="cache/profiles.json")
    ap.add_argument("--log", default="cache/phenomizer_resolution.tsv")
    a = ap.parse_args()
    onto = Ontology(a.hpo)
    ann = Annotations(a.hpoa, onto)
    kb = set(ann.diseases)
    cases = [json.loads(l) for l in open(a.cases)]
    coh_all = cohort_profiles(cases, a.min_cases)
    coh = [p for p in coh_all if p["disease_id"] in kb]
    phz, rows, skipped = phenomizer_profiles(json.load(open(a.tables))["tables"], onto, kb)
    profiles = coh + phz
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(profiles, open(a.out, "w"))
    with open(a.log, "w") as f:
        f.write("table\tfeature\tprinted\tresult\tmethod\tsimilarity\n")
        for r in rows:
            f.write("\t".join(str(r[k]) if r[k] is not None else "" for k in ("table", "feature", "printed", "result", "method", "similarity")) + "\n")
    both = {p["disease_id"] for p in coh} & {p["disease_id"] for p in phz}
    sums = sorted(sum(f["freq"] for f in p["features"]) for p in coh)
    methods = Counter(r["method"].split(":")[0] for r in rows)
    print(f"cohort profiles (>= {a.min_cases} patients): {len(coh_all)}; in the annotation file: {len(coh)}")
    print(f"expected findings per simulated patient from cohort profiles: median {sums[len(sums) // 2]:.1f}" if sums else "no cohort profiles")
    print(f"Phenomizer profiles: {len(phz)} of {len(phz) + len(skipped)} syndromes; skipped: {skipped}")
    print(f"diseases with profiles from both sources: {len(both)}; distinct diseases overall: {len({p['disease_id'] for p in profiles})}")
    print("ID resolution:", dict(methods))
    print(f"wrote {a.out} and {a.log} (every repair and drop is in the log)")


if __name__ == "__main__":
    main()
