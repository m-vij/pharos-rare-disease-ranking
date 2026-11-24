import textwrap

from src.ontology import Ontology
from src.resolve_ids import Resolver, name_parts

OBO = textwrap.dedent("""\
    data-version: hp/releases/test

    [Term]
    id: HP:0000118
    name: Phenotypic abnormality

    [Term]
    id: HP:0000015
    name: Bladder diverticulum
    synonym: "Bladder diverticula" EXACT []
    is_a: HP:0000118

    [Term]
    id: HP:0000081
    name: Duplicated collecting system
    is_a: HP:0000118

    [Term]
    id: HP:0000032
    name: Abnormality of male external genitalia
    is_a: HP:0000118

    [Term]
    id: HP:0000327
    name: Hypoplasia of the maxilla
    synonym: "Hypoplastic maxilla" EXACT []
    is_a: HP:0000118

    [Term]
    id: HP:0000189
    name: Narrow palate
    is_a: HP:0000118

    [Term]
    id: HP:0001680
    name: Coarctation of aorta
    is_a: HP:0000118

    [Term]
    id: HP:0002623
    name: Overriding aorta
    is_a: HP:0000118

    [Term]
    id: HP:0000160
    name: Narrow mouth
    synonym: "Small opening of mouth" EXACT []
    is_a: HP:0000118

    [Term]
    id: HP:0000770
    name: Foo bar
    is_a: HP:0000118

    [Term]
    id: HP:0000771
    name: Foo bar
    is_a: HP:0000118
    """)


def resolver(tmp_path):
    p = tmp_path / "hp.obo"
    p.write_text(OBO)
    o = Ontology(str(p))
    return o, Resolver(o)


def test_name_parts():
    assert name_parts("Renal agenesis and/or Renal aplasia and Hydronephrosis or Cyst") == ["Renal agenesis", "Renal aplasia", "Hydronephrosis", "Cyst"]
    assert name_parts("Seizures") == ["Seizures"]


def test_valid_ids_are_used_as_printed(tmp_path):
    o, r = resolver(tmp_path)
    ids, log = r.resolve_feature("Narrow palate", ["HP:0000189"])
    assert ids == ["HP:0000189"] and log[0]["method"] == "as printed"


def test_repairs_choose_by_name_not_just_by_shape(tmp_path):
    o, r = resolver(tmp_path)
    assert r.resolve_feature("Coarctation of aorta", ["HP:001680"])[0] == ["HP:0001680"]          # too short: zero-padded
    assert r.resolve_feature("overriding aorta", ["HP:00002623"])[0] == ["HP:0002623"]            # extra leading zero
    # 'HP:000032' zero-pads to a real but WRONG term (HP:0000032); the name picks the maxilla term with a missing last digit
    ids, log = r.resolve_feature("Hypoplastic maxilla and Narrow palate", ["HP:000032", "HP:0000189"])
    assert ids == ["HP:0000327", "HP:0000189"] and log[0]["method"] == "repaired: missing last digit"
    ids, _ = r.resolve_feature("Renal agenesis and/or bladder diverticula and/or Duplicated collecting system", ["HP:000015", "HP:000081"])
    assert ids == ["HP:0000015", "HP:0000081"]                                                    # the name part that matches is used


def test_bad_and_ambiguous_ids_are_dropped_and_logged(tmp_path):
    o, r = resolver(tmp_path)
    ids, log = r.resolve_feature("Totally unrelated words", ["HP:9999999"])
    assert ids == [] and log[0]["method"].startswith("dropped") and log[0]["result"] is None
    ids, log = r.resolve_feature("Foo bar", ["HP:000077"])                                         # HP:0000770 and HP:0000771 fit equally well
    assert ids == [] and log[0]["method"].startswith("dropped")


def test_row_without_an_id_is_matched_by_exact_name_when_unique(tmp_path):
    o, r = resolver(tmp_path)
    ids, log = r.resolve_feature("Small opening of mouth", [])
    assert ids == ["HP:0000160"] and log[0]["method"] == "matched by exact name"
    assert r.resolve_feature("Foo bar", [])[0] == []                                              # two terms share the name: no guess
    assert r.resolve_feature("Nothing like this", [])[0] == []
    assert o.dropped == {}                                                                        # repairs never pollute the ontology's counters
