import json

from src.ontology import Ontology
from src.phenopackets import load_all, parse_phenopacket
from tests.test_ontology import OBO


def onto(tmp_path):
    (tmp_path / "hp.obo").write_text(OBO)
    return Ontology(str(tmp_path / "hp.obo"))


def packet(pid, features, disease=("OMIM:1", "Disease one"), pmid="PMID:111", genes=("GENE1",), via="interpretations"):
    d = {"id": pid, "subject": {"sex": "FEMALE"},
         "phenotypicFeatures": [{"type": {"id": t, "label": "x"}, **({"excluded": True} if ex else {})} for t, ex in features],
         "metaData": {"externalReferences": [{"id": pmid, "description": "paper"}]}}
    dis = {"id": disease[0], "label": disease[1]}
    if via == "interpretations":
        d["interpretations"] = [{"diagnosis": {"disease": dis, "genomicInterpretations": [
            {"variantInterpretation": {"variationDescriptor": {"geneContext": {"symbol": g}}}} for g in genes]}}]
    else:
        d["diseases"] = [{"term": dis}]
    return d


def test_parse_one_patient(tmp_path):
    o = onto(tmp_path)
    d = packet("PMID_111_p1", [("HP:0000099", False), ("HP:0000012", False), ("HP:0000010", True), ("HP:9999999", False), ("HP:0000005", False)])
    rec, why = parse_phenopacket(d, o, "cohortA")
    assert why is None
    assert rec["present"] == ["HP:0000011", "HP:0000012"]                  # alternative ID normalised; unknown and non-phenotypic terms dropped
    assert rec["absent"] == ["HP:0000010"] and rec["n_remapped"] == 1
    assert (rec["disease_id"], rec["pmids"], rec["genes"], rec["sex"], rec["cohort"]) == ("OMIM:1", ["PMID:111"], ["GENE1"], "FEMALE", "cohortA")


def test_a_term_both_present_and_excluded_counts_as_present_and_other_fallbacks(tmp_path):
    o = onto(tmp_path)
    d = packet("PMID_222_p2", [("HP:0000010", False), ("HP:0000010", True)], via="diseases", pmid="not-a-pmid")
    d["metaData"] = {}
    rec, _ = parse_phenopacket(d, o, "c")
    assert rec["present"] == ["HP:0000010"] and rec["absent"] == []        # excluded cannot override present
    assert rec["pmids"] == ["PMID:222"] and rec["disease_id"] == "OMIM:1"   # disease from the diseases list; PMID from the case id


def test_drop_reasons_and_report(tmp_path):
    o = onto(tmp_path)
    root = tmp_path / "ps"
    (root / "cohortA" / "phenopackets").mkdir(parents=True)
    (root / "cohortB").mkdir()
    good = [packet(f"PMID_1_{i}", [("HP:0000011", False)]) for i in range(3)]
    for i, d in enumerate(good):
        (root / "cohortA" / "phenopackets" / f"{i}.json").write_text(json.dumps(d))
    (root / "cohortB" / "nofind.json").write_text(json.dumps(packet("PMID_2_1", [("HP:9999999", False)])))
    (root / "cohortB" / "noomim.json").write_text(json.dumps(packet("PMID_2_2", [("HP:0000012", False)], disease=("ORPHA:5", "x"))))
    (root / "cohortB" / "other.json").write_text(json.dumps({"hello": 1}))
    (root / "cohortB" / "broken.json").write_text("{not json")
    records, rep = load_all(root, o)
    assert rep["files"] == 7 and rep["cases"] == 3
    assert rep["dropped"] == {"no findings": 1, "no OMIM disease": 1, "not a phenopacket": 1, "unreadable file": 1}
    assert rep["diseases"] == 1 and rep["cohorts"] == 1 and rep["with_gene"] == 3 and rep["with_exactly_one_pmid"] == 3
    assert rep["diseases_with_1_case"] == 0 and {r["cohort"] for r in records} == {"cohortA"}
