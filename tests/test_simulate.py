import json
from collections import Counter

import numpy as np

from src.simulate import run, simulate, simulate_profile


def profile(pid, features, pmids=(), source="phenopackets"):
    return {"id": pid, "disease_id": pid.split("|")[0], "name": "Disease " + pid, "source": source, "n_cases": None,
            "pmids": list(pmids), "features": features}


def feat(terms, freq, op="single", sex=None):
    return {"terms": terms, "op": op, "freq": freq, "sex": sex}


def test_features_are_drawn_at_their_frequencies():
    p = profile("OMIM:1|phenopackets", [feat(["A"], 1.0), feat(["B"], 0.3), feat(["C"], 0.7)])      # A always, so no patient is empty
    cases = simulate_profile(p, 4000, seed=1)
    sets = [set(c["present"]) for c in cases]
    assert all("A" in s for s in sets)
    assert abs(np.mean(["B" in s for s in sets]) - 0.3) < 0.025 and abs(np.mean(["C" in s for s in sets]) - 0.7) < 0.025
    assert abs(np.mean([("B" in s and "C" in s) for s in sets]) - 0.21) < 0.025                         # independent: 0.3 * 0.7


def test_operators_for_features_with_several_terms():
    one = lambda op: simulate_profile(profile("OMIM:2|phenomizer", [feat(["X", "Y", "Z"], 1.0, op)], source="phenomizer"), 1200, seed=2)
    assert {tuple(c["present"]) for c in one("and")} == {("X", "Y", "Z")}                               # all together
    ors = one("or")
    assert {len(c["present"]) for c in ors} == {1} and all(abs(n / 1200 - 1 / 3) < 0.05 for n in Counter(c["present"][0] for c in ors).values())
    andor = one("andor")
    assert all(1 <= len(c["present"]) <= 3 for c in andor)                                              # never empty
    assert len({tuple(c["present"]) for c in andor}) == 7                                               # every non-empty subset occurs
    assert abs(np.mean([len(c["present"]) for c in andor]) - 1.625) < 0.06                              # 1.5 + one extra term in the 1/8 empty draws


def test_sex_specific_features_and_the_sex_split():
    p = profile("OMIM:3|phenopackets", [feat(["ANY"], 1.0), feat(["MALE_ONLY"], 1.0, sex="male"), feat(["FEMALE_ONLY"], 1.0, sex="female")])
    cases = simulate_profile(p, 1000, seed=3)
    for c in cases:
        assert ("MALE_ONLY" in c["present"]) == (c["sex"] == "male") and ("FEMALE_ONLY" in c["present"]) == (c["sex"] == "female")
    assert abs(np.mean([c["sex"] == "male" for c in cases]) - 0.5) < 0.05


def test_never_empty_and_the_fallback_uses_the_most_frequent_feature():
    p = profile("OMIM:4|phenopackets", [feat(["RARE1"], 0.0001), feat(["RARE2"], 0.0003)])
    cases = simulate_profile(p, 50, seed=4)
    assert len(cases) == 50 and all(c["present"] == ["RARE2"] for c in cases)                          # fallback after 20 empty draws
    only_male = profile("OMIM:5|phenopackets", [feat(["M"], 0.0001, sex="male")])
    assert {c["sex"] for c in simulate_profile(only_male, 40, seed=5)} == {"male"}                      # a female patient has nothing to show: skipped


def test_reproducible_independent_of_other_profiles_and_records_the_papers():
    a = profile("OMIM:1|phenopackets", [feat(["A"], 0.5), feat(["B"], 0.5)], pmids=["PMID:1", "PMID:2"])
    b = profile("OMIM:2|phenomizer", [feat(["C"], 0.5), feat(["D"], 0.5)], source="phenomizer")
    both = simulate([a, b], per_profile=10, seed=7)
    assert [c for c in both if c["profile_id"] == a["id"]] == simulate([a], per_profile=10, seed=7)     # adding a profile changes nothing for the others
    assert both != simulate([a, b], per_profile=10, seed=8)
    first = both[0]
    assert first["exclude_pmids"] == ["PMID:1", "PMID:2"] and first["disease_id"] == "OMIM:1" and first["source"] == "phenopackets"
    assert len({c["case_id"] for c in both}) == 20 and [c for c in both if c["source"] == "phenomizer"][0]["exclude_pmids"] == []
    assert {c["profile_id"] for c in simulate([a, b], 5, 0, sources=("phenomizer",))} == {b["id"]}


def test_run_writes_cases_and_a_summary(tmp_path):
    profs = [profile("OMIM:1|phenopackets", [feat(["A"], 1.0), feat(["B"], 0.5)], pmids=["PMID:1"]),
             profile("OMIM:2|phenomizer", [feat(["C"], 0.9)], source="phenomizer")]
    (tmp_path / "profiles.json").write_text(json.dumps(profs))
    cases, summary = run(str(tmp_path / "profiles.json"), str(tmp_path / "out" / "sim.jsonl"), 20, 0, ["phenopackets", "phenomizer"])
    lines = [json.loads(l) for l in open(tmp_path / "out" / "sim.jsonl")]
    assert len(lines) == len(cases) == 40
    assert summary["phenopackets"]["patients"] == 20 and summary["phenopackets"]["patients_with_exclusions"] == 20
    assert summary["phenopackets"]["median_expected_findings"] == 1.5 and summary["phenomizer"]["patients_with_exclusions"] == 0
