import textwrap

from src.ontology import Ontology

OBO = textwrap.dedent("""\
    format-version: 1.2
    data-version: hp/releases/2026-09-01

    [Term]
    id: HP:0000001
    name: All

    [Term]
    id: HP:0000118
    name: Phenotypic abnormality
    is_a: HP:0000001 ! All

    [Term]
    id: HP:0000005
    name: Mode of inheritance
    is_a: HP:0000001 ! All

    [Term]
    id: HP:0000010
    name: Seizure
    is_a: HP:0000118 ! Phenotypic abnormality
    synonym: "Fits" EXACT [HPO:probinson]
    synonym: "Convulsion" RELATED []

    [Term]
    id: HP:0000011
    name: Epileptic spasm
    alt_id: HP:0000099
    is_a: HP:0000010 {source="x"} ! Seizure

    [Term]
    id: HP:0000012
    name: Delay
    is_a: HP:0000118

    [Term]
    id: HP:0000013
    name: Spasm with delay
    is_a: HP:0000011
    is_a: HP:0000012

    [Term]
    id: HP:0000020
    name: Old name
    is_obsolete: true
    replaced_by: HP:0000011

    [Term]
    id: HP:0000021
    name: Obsolete, nothing replaces it
    is_obsolete: true

    [Typedef]
    id: part_of
    name: part of
    """)


def onto(tmp_path):
    p = tmp_path / "hp.obo"
    p.write_text(OBO)
    return Ontology(str(p))


def test_structure(tmp_path):
    o = onto(tmp_path)
    assert o.version == "hp/releases/2026-09-01" and o.name("HP:0000010") == "Seizure"
    assert o.ancestors("HP:0000013") == {"HP:0000011", "HP:0000012", "HP:0000010", "HP:0000118", "HP:0000001"}   # two parents; modifier on is_a ignored
    assert o.with_ancestors("HP:0000010") == {"HP:0000010", "HP:0000118", "HP:0000001"}
    assert o.descendants("HP:0000010") == {"HP:0000011", "HP:0000013"}
    assert o.is_phenotypic("HP:0000013") and o.is_phenotypic("HP:0000118") and not o.is_phenotypic("HP:0000005")


def test_normalize_follows_alt_and_replacement_and_counts_failures(tmp_path):
    o = onto(tmp_path)
    assert o.normalize("HP:0000099") == "HP:0000011"                 # alternative ID
    assert o.normalize(" HP:0000020 ") == "HP:0000011"               # obsolete with a replacement
    assert o.normalize("HP:0000010") == "HP:0000010"
    assert o.normalize("HP:0000021") is None and o.normalize("HP:9999999") is None and o.normalize("") is None
    assert o.dropped == {"obsolete, no replacement": 1, "unknown id": 2}


def test_synonyms_and_stats(tmp_path):
    o = onto(tmp_path)
    assert o.synonyms("HP:0000010") == ["Fits"] and o.synonyms("HP:0000010", ("EXACT", "RELATED")) == ["Fits", "Convulsion"]
    s = o.stats()
    assert s["terms"] == 7 and s["obsolete"] == 2 and s["alt ids"] == 1 and s["phenotypic terms"] == 5     # obsolete terms are not counted as terms
