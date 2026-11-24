import numpy as np

from src.evaluate import bootstrap, evaluate, harmonic, rank_stats
from src.scorer import make_scorer
from tests.test_scorer import kb


def test_rank_stats_with_ties_by_expected_value():
    H = harmonic(4)
    m = rank_stats(np.array([1.0, 0.0, 1.0, 0.0]), 2, H, ks=(1, 2, 5))         # true disease ties for first with disease 0
    assert abs(m["R@1"] - 0.5) < 1e-9 and m["R@2"] == 1.0 and m["R@5"] == 1.0 and abs(m["MRR"] - 0.75) < 1e-9
    m = rank_stats(np.array([3.0, 2.0, 1.0, 0.0]), 3, H, ks=(1, 2, 5))         # last of four
    assert m["R@1"] == 0.0 and m["R@2"] == 0.0 and m["R@5"] == 1.0 and abs(m["MRR"] - 0.25) < 1e-9


def test_bootstrap_interval_brackets_the_mean_and_collapses_when_all_equal():
    mean, lo, hi = bootstrap([0.0, 1.0, 1.0, 0.0, 1.0, 1.0], 500)
    assert abs(mean - 4 / 6) < 1e-9 and lo <= mean <= hi and hi > lo
    assert bootstrap([0.5, 0.5, 0.5], 100) == (0.5, 0.5, 0.5)


def case(cid, disease, present, source="phenopackets", pmids=()):
    return {"case_id": cid, "profile_id": f"{disease}|{source}", "disease_id": disease, "source": source,
            "present": present, "exclude_pmids": list(pmids)}


def test_evaluate_by_hand_with_per_disease_averaging_and_the_paper_removal(tmp_path):
    ann, _ = kb(tmp_path)
    sc = [make_scorer("exact_overlap", ann)]
    cases = [case("a", "OMIM:3", ["HP:0000011"]),                                    # exact scores [1,0,1,0]: true D3 ties with D1 -> R@1 .5, MRR .75
             case("b", "OMIM:4", ["HP:0000013"], pmids=["PMID:555"]),               # D4's only row removed: every disease scores 0 -> R@1 .25, MRR 0.5208
             case("c", "OMIM:3", ["HP:0000011"]),                                    # second patient of D3: the disease still counts once
             case("d", "OMIM:3", ["HP:0000011"], source="phenomizer")]               # same disease, other source: reported separately
    res = evaluate(ann, sc, cases, n_boot=50, remove_papers=True)["exact_overlap"]
    p = res["phenopackets"]
    assert p["diseases"] == 2 and p["patients"] == 3
    assert abs(p["R@1"][0] - (0.5 + 0.25) / 2) < 1e-9 and abs(p["MRR"][0] - (0.75 + (1 + 0.5 + 1 / 3 + 0.25) / 4) / 2) < 1e-9
    assert res["phenomizer"]["diseases"] == 1 and abs(res["phenomizer"]["R@1"][0] - 0.5) < 1e-9


def test_by_default_a_simulated_patient_is_scored_against_the_full_knowledge_base(tmp_path):
    ann, _ = kb(tmp_path)
    sc = [make_scorer("exact_overlap", ann)]
    b = case("b", "OMIM:4", ["HP:0000013"], pmids=["PMID:555"])                      # the paper list is ignored unless asked for
    full = evaluate(ann, sc, [b], n_boot=20)["exact_overlap"]["phenopackets"]
    assert full["R@1"][0] == 1.0 and full["MRR"][0] == 1.0                           # D4 still has its row, so it ranks first alone
    removed = evaluate(ann, sc, [b], n_boot=20, remove_papers=True)["exact_overlap"]["phenopackets"]
    assert removed["R@1"][0] == 0.25
