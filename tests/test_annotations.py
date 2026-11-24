from src.annotations import Annotations
from src.ontology import Ontology
from tests.test_ontology import OBO

HEADER = "database_id\tdisease_name\tqualifier\thpo_id\treference\tevidence\tonset\tfrequency\tsex\tmodifier\taspect\tbiocuration"


def row(db, name, qual, hpo, ref, aspect="P"):
    return "\t".join([db, name, qual, hpo, ref, "PCS", "", "", "", "", aspect, "x"])


def build(tmp_path, rows):
    (tmp_path / "hp.obo").write_text(OBO)
    (tmp_path / "p.hpoa").write_text("#description: test\n#version: 2026-09-02\n" + HEADER + "\n" + "\n".join(rows) + "\n")
    onto = Ontology(str(tmp_path / "hp.obo"))
    return onto, Annotations(str(tmp_path / "p.hpoa"), onto)


ROWS = [
    row("OMIM:1", "Disease one", "", "HP:0000011", "PMID:111;OMIM:1"),
    row("OMIM:1", "Disease one", "", "HP:0000012", "PMID:222, PMID:333"),
    row("OMIM:1", "Disease one", "NOT", "HP:0000010", "PMID:444"),            # explicit absence: not a positive annotation
    row("OMIM:1", "Disease one", "", "HP:0000005", "PMID:555", aspect="I"),    # inheritance row
    row("OMIM:2", "Disease two", "", "HP:0000099", "OMIM:2"),                  # alternative ID of the epileptic spasm term
    row("OMIM:2", "Disease two", "", "HP:9999999", "OMIM:2"),                  # unknown term
    row("OMIM:3", "Disease three", "", "HP:0000005", "OMIM:3"),                # not a phenotypic abnormality
    row("ORPHA:4", "Disease four", "", "HP:0000011", "ORPHA:4"),               # other prefix
]


def test_reading_rules(tmp_path):
    onto, ann = build(tmp_path, ROWS)
    assert ann.version == "2026-09-02"
    assert ann.diseases == ["OMIM:1", "OMIM:2"]
    assert ann.pos["OMIM:1"] == {"HP:0000011", "HP:0000012"} and ann.neg["OMIM:1"] == {"HP:0000010"}
    assert ann.pos["OMIM:2"] == {"HP:0000011"}                                 # normalised from the alternative ID
    assert ann.names["OMIM:1"] == "Disease one"
    assert ann.report["rows"] == 8 and ann.report["rows_not_aspect_P"] == 1 and ann.report["rows_other_prefix"] == 1
    assert ann.report["rows_unresolvable_term"] == 1 and ann.report["rows_non_phenotypic"] == 1 and ann.report["rows_NOT"] == 1


def test_references_are_split_and_kept(tmp_path):
    onto, ann = build(tmp_path, ROWS)
    assert ann.refs[("OMIM:1", "HP:0000011")] == {"PMID:111", "OMIM:1"}
    assert ann.refs[("OMIM:1", "HP:0000012")] == {"PMID:222", "PMID:333"}


def test_closure_adds_phenotypic_ancestors_only(tmp_path):
    onto, ann = build(tmp_path, ROWS)
    # spasm -> seizure -> phenotypic abnormality; delay -> phenotypic abnormality. 'All' is not phenotypic, so it is left out
    assert ann.closure("OMIM:1") == {"HP:0000011", "HP:0000010", "HP:0000012", "HP:0000118"}
    assert ann.closure("OMIM:99") == frozenset()
    assert ann.stats()["candidate diseases"] == 2
