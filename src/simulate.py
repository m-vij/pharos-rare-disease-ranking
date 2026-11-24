"""Simulated patients drawn from disease profiles (the Phenomizer procedure, Kohler et al. 2009).

For one simulated patient of a disease:
  1. decide the sex (50/50). Features marked male-only or female-only apply only to that sex.
  2. for each feature, draw a random number; the feature is present if it is below the feature's frequency
     (features are independent, which the paper itself notes is not true but cannot be modelled without more data).
  3. a feature that lists several HPO terms is realised by its operator (the supplement does not define these, so these
     are my rules): 'and' = all the terms together; 'or' = exactly one, chosen uniformly; 'and/or' = a random non-empty
     subset (each term kept with probability 1/2, one chosen if none was).
  4. a patient needs at least one finding: if none was drawn the draw is repeated (up to 20 times), which nudges the
     frequencies of diseases with very few features upward, then falls back to the most frequent applicable feature.

Each profile has its own random stream (seeded from the seed and the profile ID), so adding or removing profiles never
changes the patients of the others. Every patient records the papers (PMIDs) its profile came from, so the evaluation can
remove them from the knowledge base.

    python -m src.simulate --per-profile 30
"""
import argparse
import json
import random
from pathlib import Path


def _realise(feature, rng):
    terms = feature["terms"]
    op = feature["op"] if len(terms) > 1 else "single"
    if op in ("single", "and"):
        return list(terms)
    if op == "or":
        return [rng.choice(terms)]
    kept = [t for t in terms if rng.random() < 0.5]                          # 'andor'
    return kept or [rng.choice(terms)]


def _draw(features, sex, rng):
    found = []
    for f in features:
        if f["sex"] and f["sex"] != sex:
            continue
        if rng.random() < f["freq"]:
            found += _realise(f, rng)
    return sorted(set(found))


def simulate_profile(profile, n, seed=0):
    rng = random.Random(f"{seed}|{profile['id']}")
    out = []
    for i in range(n):
        sex = rng.choice(["male", "female"])
        present = []
        for _ in range(20):
            present = _draw(profile["features"], sex, rng)
            if present:
                break
        if not present:
            usable = [f for f in profile["features"] if not f["sex"] or f["sex"] == sex]
            if not usable:
                continue
            present = sorted(set(_realise(max(usable, key=lambda f: (f["freq"], f["terms"])), rng)))
        out.append({"case_id": "SIM_" + profile["id"].replace(":", "").replace("|", "_") + f"_{i}", "profile_id": profile["id"],
                    "disease_id": profile["disease_id"], "disease_label": profile["name"], "source": profile["source"],
                    "sex": sex, "present": present, "exclude_pmids": list(profile["pmids"])})
    return out


def simulate(profiles, per_profile=30, seed=0, sources=("phenopackets", "phenomizer")):
    cases = []
    for p in profiles:
        if p["source"] in sources and p["features"]:
            cases += simulate_profile(p, per_profile, seed)
    return cases


def run(profiles_path, out_path, per_profile, seed, sources):
    profiles = json.load(open(profiles_path))
    cases = simulate(profiles, per_profile, seed, sources)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for c in cases:
            f.write(json.dumps(c) + "\n")
    summary = {}
    for src in sources:
        cs = [c for c in cases if c["source"] == src]
        if cs:
            sizes = sorted(len(c["present"]) for c in cs)
            exp = sorted(sum(f["freq"] for f in p["features"]) for p in profiles if p["source"] == src and p["features"])
            summary[src] = {"profiles": len({c["profile_id"] for c in cs}), "patients": len(cs), "median_findings": sizes[len(sizes) // 2],
                            "max_findings": sizes[-1], "median_expected_findings": round(exp[len(exp) // 2], 1),
                            "patients_with_exclusions": sum(1 for c in cs if c["exclude_pmids"])}
    return cases, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default="cache/profiles.json")
    ap.add_argument("--out", default="cache/sim_cases.jsonl")
    ap.add_argument("--per-profile", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", default="phenopackets,phenomizer")
    a = ap.parse_args()
    _, summary = run(a.profiles, a.out, a.per_profile, a.seed, a.sources.split(","))
    for src, s in summary.items():
        print(f"{src}: {s}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
