import math

import numpy as np

from src.scorer import SCORERS, Rarity, make_scorer
from tests.test_annotations import build, row

L43, L4 = math.log(4 / 3), math.log(4)
ROWS = [row("OMIM:1", "D1", "", "HP:0000011", "PMID:111"), row("OMIM:1", "D1", "", "HP:0000012", "PMID:222"),
        row("OMIM:2", "D2", "", "HP:0000012", "PMID:333"),
        row("OMIM:3", "D3", "", "HP:0000011", "PMID:444"),
        row("OMIM:4", "D4", "", "HP:0000013", "PMID:555")]
SPASM, DELAY, SPASM_DELAY = "HP:0000011", "HP:0000012", "HP:0000013"      # spasm -> seizure; spasm_delay is below spasm AND delay


def kb(tmp_path):
    onto, ann = build(tmp_path, ROWS)
    return ann, ann.view(())


def close(a, b):
    return np.allclose(a, b, atol=1e-8)


def test_rarity(tmp_path):
    ann, _ = kb(tmp_path)
    r = Rarity(ann)
    # closures: D1 {spasm, seizure, delay, root}, D2 {delay, root}, D3 {spasm, seizure, root}, D4 {all five}
    assert r.n == 4 and abs(r.ic("HP:0000118")) < 1e-12 and abs(r.ic(SPASM) - L43) < 1e-12 and abs(r.ic(SPASM_DELAY) - L4) < 1e-12
    assert abs(r.ic("HP:9999999") - L4) < 1e-12                                # an unseen term counts as the rarest possible


def test_scores_by_hand(tmp_path):
    ann, v = kb(tmp_path)
    s = {n: make_scorer(n, ann) for n in SCORERS}
    assert close(s["annotation_count"].score([SPASM], v), [2, 1, 1, 1])
    assert close(s["exact_overlap"].score([SPASM], v), [1, 0, 1, 0])           # only D1 and D3 list spasm directly
    assert close(s["hierarchy_overlap"].score([SPASM], v), [2, 0, 2, 2])        # shared with spasm + seizure: D1, D3 and D4 (below spasm); D2 none
    assert close(s["resnik_best_match"].score([SPASM], v), [L43, 0, L43, L43])
    assert close(s["exact_overlap"].score([SPASM_DELAY], v), [0, 0, 0, 1])
    assert close(s["hierarchy_overlap"].score([SPASM_DELAY], v), [3, 1, 2, 4])   # patient closure {spasm_delay, spasm, delay, seizure}
    assert close(s["resnik_best_match"].score([SPASM_DELAY], v), [L43, L43, L43, L4])    # only D4 shares the rare term itself
    assert close(s["resnik_best_match"].score([SPASM, DELAY], v), [2 * L43, L43, L43, 2 * L43])


def test_the_view_changes_only_the_diseases_whose_rows_it_removed(tmp_path):
    ann, _ = kb(tmp_path)
    v = ann.view({"PMID:555"})                                                  # the only paper behind D4's annotation
    assert close(make_scorer("resnik_best_match", ann).score([SPASM_DELAY], v), [L43, L43, L43, 0])
    assert close(make_scorer("exact_overlap", ann).score([SPASM_DELAY], v), [0, 0, 0, 0])
    assert close(make_scorer("annotation_count", ann).score([SPASM], v), [2, 1, 1, 0])


def test_vectorised_scores_equal_the_explicit_ones_for_every_disease(tmp_path):
    ann, v = kb(tmp_path)
    for name in SCORERS[1:]:
        sc = make_scorer(name, ann)
        for present in ([SPASM], [SPASM_DELAY], [SPASM, DELAY], [DELAY, SPASM_DELAY]):
            base = sc._base(sorted(present))
            for i, d in enumerate(ann.diseases):
                assert abs(base[i] - sc._one(sorted(present), v.pos(d), v.closure(d))) < 1e-9, (name, present, d)


def test_random_is_fixed_per_patient_and_ignores_the_view(tmp_path):
    ann, v = kb(tmp_path)
    r = make_scorer("random", ann, seed=1)
    a = r.score([SPASM], v)
    assert np.array_equal(a, r.score([SPASM], ann.view({"PMID:555"}))) and not np.array_equal(a, r.score([DELAY], v))
    assert not np.array_equal(a, make_scorer("random", ann, seed=2).score([SPASM], v))
