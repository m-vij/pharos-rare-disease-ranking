from tests.test_annotations import ROWS, build


def test_view_removes_rows_citing_the_papers(tmp_path):
    onto, ann = build(tmp_path, ROWS)
    v = ann.view({"PMID:111"})                                              # supplied the spasm row of disease one
    assert v.overrides == {"OMIM:1": frozenset({"HP:0000012"})}
    assert v.pos("OMIM:1") == {"HP:0000012"} and v.pos("OMIM:2") == {"HP:0000011"}      # other diseases untouched
    assert v.closure("OMIM:1") == {"HP:0000012", "HP:0000118"} and v.closure("OMIM:2") == ann.closure("OMIM:2")
    assert abs(v.removed_fraction("OMIM:1") - 0.5) < 1e-9 and v.removed_fraction("OMIM:2") == 0.0
    assert ann.pos["OMIM:1"] == {"HP:0000011", "HP:0000012"}                # the base is never modified


def test_view_with_several_papers_nothing_and_everything(tmp_path):
    onto, ann = build(tmp_path, ROWS)
    assert ann.view(()).overrides == {} and ann.view({"PMID:999"}).overrides == {}
    v = ann.view({"PMID:111", "PMID:222"})
    assert v.pos("OMIM:1") == frozenset() and v.removed_fraction("OMIM:1") == 1.0
    assert v.closure("OMIM:1") == frozenset()
    assert ann.view({"OMIM:2"}).overrides == {}                              # only PMID references count as papers
