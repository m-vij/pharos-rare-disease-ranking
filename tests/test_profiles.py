from src.ontology import Ontology
from src.profiles import cohort_profiles, phenomizer_profiles
from tests.test_resolve_ids import OBO


def case(cid, disease, present, pmids=("PMID:1",), label="Disease one"):
    return {"case_id": cid, "disease_id": disease, "disease_label": label, "present": present, "pmids": list(pmids)}


def test_cohort_profile_frequencies_need_enough_patients():
    cases = [case(f"a{i}", "OMIM:1", ["T1", "T2"] if i < 5 else ["T1"], pmids=("PMID:1",) if i < 7 else ("PMID:2",)) for i in range(10)]
    cases += [case(f"b{i}", "OMIM:2", ["T9"]) for i in range(9)]                      # nine patients: below the minimum
    profs = cohort_profiles(cases, min_cases=10)
    assert [p["disease_id"] for p in profs] == ["OMIM:1"]
    p = profs[0]
    assert (p["source"], p["n_cases"], p["pmids"], p["id"]) == ("phenopackets", 10, ["PMID:1", "PMID:2"], "OMIM:1|phenopackets")
    assert [(f["terms"], f["freq"]) for f in p["features"]] == [(["T1"], 1.0), (["T2"], 0.5)]
    assert abs(sum(f["freq"] for f in p["features"]) - 1.5) < 1e-9                    # expected findings per simulated patient = mean real patient size


def test_phenomizer_profiles_repair_ids_and_skip_unusable_syndromes(tmp_path):
    (tmp_path / "hp.obo").write_text(OBO)
    onto = Ontology(str(tmp_path / "hp.obo"))
    feat = lambda name, ids, op, freq: {"name": name, "ids": ids, "op": op, "freq": freq, "sex": None}
    tables = [
        {"table": 2, "name": "Syndrome A", "omim": "OMIM:1", "features": [
            feat("Narrow palate", ["HP:0000189"], "single", 0.5),
            feat("Coarctation of aorta", ["HP:001680"], "single", 0.25),                    # repaired
            feat("Hypoplastic maxilla and Narrow palate", ["HP:000032", "HP:0000189"], "and", 0.9),
            feat("Totally unrelated", ["HP:9999999"], "single", 0.3),                       # unusable: dropped
            feat("Narrow mouth", [], "single", None)]},                                     # no id and no frequency
        {"table": 3, "name": "Syndrome B", "omim": None, "features": []},
        {"table": 4, "name": "Syndrome C", "omim": "OMIM:3", "features": []}]
    profs, rows, skipped = phenomizer_profiles(tables, onto, {"OMIM:1"})
    assert [p["disease_id"] for p in profs] == ["OMIM:1"] and len(skipped) == 2
    assert [(f["terms"], f["op"], f["freq"]) for f in profs[0]["features"]] == [
        (["HP:0000189"], "single", 0.5), (["HP:0001680"], "single", 0.25), (["HP:0000327", "HP:0000189"], "and", 0.9)]
    assert any(r["method"].startswith("repaired") for r in rows) and any(r["method"].startswith("feature dropped") for r in rows)
