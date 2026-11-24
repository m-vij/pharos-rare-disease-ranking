import pytest

from src.names import NameIndex, normalize
from src.ontology import Ontology
from src.prompt import (build_alone_prompt, build_grammar, build_rag_prompt, check_no_leak, parse_free_text,
                        parse_ranking, unique_names)
from tests.test_ontology import OBO


def onto(tmp_path):
    (tmp_path / "hp.obo").write_text(OBO)
    return Ontology(str(tmp_path / "hp.obo"))


CANDS = [{"disease_id": "OMIM:3", "name": "Alpha"}, {"disease_id": "OMIM:1", "name": "Alpha"}, {"disease_id": "OMIM:2", "name": "Beta"}]


def test_names_normalisation_and_exact_matching():
    assert normalize("Cohen syndrome, type II (rare)") == "cohen syndrome type 2" and normalize("  KABUKI  syndrome 1 ") == "kabuki syndrome 1"
    ix = NameIndex({"OMIM:1": "Noonan syndrome 1", "OMIM:2": "Noonan syndrome 2", "OMIM:3": "Twin", "OMIM:4": "Twin"})
    assert ix.match("noonan SYNDROME 2") == {"OMIM:2"} and ix.match("Noonan syndrome") == set() and ix.match("twin") == {"OMIM:3", "OMIM:4"}


def test_unique_names_and_a_shuffle_that_ignores_input_order(tmp_path):
    assert unique_names(CANDS) == {"OMIM:1": "Alpha", "OMIM:3": "Alpha (2)", "OMIM:2": "Beta"}
    o = onto(tmp_path)
    a = build_rag_prompt(["HP:0000011", "HP:0000012"], o, CANDS, seed=1)
    b = build_rag_prompt(["HP:0000012", "HP:0000011"], o, list(reversed(CANDS)), seed=1)
    assert a["messages"] == b["messages"] and a["name_to_id"] == {"Alpha": "OMIM:1", "Alpha (2)": "OMIM:3", "Beta": "OMIM:2"}
    assert a["messages"][1]["content"].startswith("Findings present:\n- Delay\n- Epileptic spasm\n\nCandidate diseases:\n- ")
    assert "HP:" not in a["messages"][1]["content"] and "OMIM" not in a["messages"][1]["content"]
    orders = {tuple(build_rag_prompt(["HP:0000011"], o, CANDS, seed=s)["candidate_names"]) for s in range(20)}
    assert len(orders) > 1                                                            # the seed really changes the order


def test_alone_prompt_has_no_candidates(tmp_path):
    p = build_alone_prompt(["HP:0000011"], onto(tmp_path), 5)
    assert "Candidate" not in p["messages"][1]["content"] and "up to 5 diseases" in p["messages"][1]["content"]


def test_grammar_shape_and_escaping():
    g = build_grammar(['Say "hi"', "A\\B", "C"], 3)
    assert g.splitlines()[:3] == ["root ::= item r1", 'r1 ::= (nl item r2)?', 'r2 ::= (nl item)?']
    assert 'item ::= "Say \\"hi\\"" | "A\\\\B" | "C"' in g and 'nl ::= "\\n"' in g
    assert build_grammar(["x"], 5).splitlines()[0] == "root ::= item"                  # never more items than candidates


def test_parse_ranking_and_free_text():
    m = {"Alpha": "OMIM:1", "Beta": "OMIM:2"}
    assert parse_ranking("1. Alpha\n- beta\nAlpha\nSomething else\n\n", m) == ([["OMIM:1"], ["OMIM:2"]], 1)     # numbering, case, repeats, unmatched
    ix = NameIndex({"OMIM:1": "Alpha", "OMIM:3": "Twin", "OMIM:4": "Twin"})
    assert parse_free_text("* alpha\ntwin\nUnknown\n", ix) == ([["OMIM:1"], ["OMIM:3", "OMIM:4"]], 1)


def test_check_no_leak():
    case = {"case_id": "SIM_OMIM1_phenopackets_0", "profile_id": "OMIM:1|phenopackets", "disease_id": "OMIM:1", "exclude_pmids": ["PMID:9"]}
    ok = [{"role": "user", "content": "Findings present:\n- Seizure\n\nCandidate diseases:\n- Alpha"}]
    check_no_leak(ok, case)
    for bad in ("SIM_OMIM1_phenopackets_0", "OMIM:1", "PMID:9", "a simulated patient SIMULATED"):
        with pytest.raises(ValueError):
            check_no_leak([{"role": "user", "content": "text " + bad}], case)
