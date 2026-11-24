import json

from src.real_eval import build_cases, evaluate_real, stratum
from src.real_llm import PatientKB, run_real
from src.scorer import make_scorer
from tests.test_run_llm import Stub, serve
from tests.test_scorer import kb


def real(cid, disease, present, pmids, label="x"):
    return {"case_id": cid, "disease_id": disease, "disease_label": label, "present": present, "pmids": pmids, "sex": "FEMALE"}


REAL = [real("PMID_111_a", "OMIM:1", ["HP:0000011"], ["PMID:111"]),        # D1 loses its spasm row (1 of 2): partly removed
        real("PMID_555_b", "OMIM:4", ["HP:0000013"], ["PMID:555"]),        # D4 loses its only row: all removed
        real("PMID_999_c", "OMIM:3", ["HP:0000011"], ["PMID:999"]),        # the paper behind nothing in the knowledge base: none removed
        real("PMID_999_d", "OMIM:3", ["HP:0000012"], ["PMID:999"]),
        real("PMID_1_e", "OMIM:77", ["HP:0000011"], ["PMID:1"]),           # a disease that is not a candidate: dropped
        real("PMID_2_f", "OMIM:2", ["HP:0000012"], ["PMID:333"])]          # D2 loses its only row: all removed


def test_strata_and_filters(tmp_path):
    ann, _ = kb(tmp_path)
    assert [stratum(x) for x in (0.0, 0.5, 1.0)] == ["real_none_removed", "real_partly_removed", "real_all_removed"]
    cs = build_cases(REAL, ann)
    assert {c["case_id"]: c["source"] for c in cs} == {"PMID_111_a": "real_partly_removed", "PMID_555_b": "real_all_removed", "PMID_999_c": "real_none_removed",
                                                       "PMID_999_d": "real_none_removed", "PMID_2_f": "real_all_removed"}
    assert all(c["exclude_pmids"] and c["profile_id"] == c["case_id"] for c in cs)
    assert "OMIM:3" not in {c["disease_id"] for c in build_cases(REAL, ann, exclude_diseases={"OMIM:3"})}        # the tuning diseases are left out
    assert [c["case_id"] for c in build_cases(REAL, ann, per_disease=1) if c["disease_id"] == "OMIM:3"] == ["PMID_999_c"]
    assert {c["source"] for c in build_cases(REAL, ann, skip_all_removed=True)} == {"real_partly_removed", "real_none_removed"}


def test_scorer_evaluation_by_hand_with_strata_and_pooling(tmp_path):
    ann, _ = kb(tmp_path)
    cs = build_cases(REAL, ann)
    res = evaluate_real(ann, [make_scorer("exact_overlap", ann)], cs, n_boot=20)["exact_overlap"]
    assert set(res) == {"real_all", "real_all_removed", "real_none_removed", "real_partly_removed"}
    assert res["real_all"]["patients"] == 5 and res["real_all"]["diseases"] == 4 and res["real_none_removed"]["patients"] == 2
    # all removed: b (D4) leaves all four diseases tied at 0 -> R@1 1/4; f (D2) leaves D1 ahead with 1 and D2, D3, D4 tied at 0 -> R@1 0; mean 0.125
    assert abs(res["real_all_removed"]["R@1"][0] - 0.125) < 1e-9
    # partly removed: patient a [spasm] against D1 {delay} (spasm row removed), D2 {delay}, D3 {spasm}, D4 {spasm_delay}: exact overlap [0,0,1,0]; D1 ties 3-way at 0
    assert abs(res["real_partly_removed"]["R@1"][0] - 0.0) < 1e-9
    # none removed: c [spasm] -> D1 and D3 tie at 1 (R@1 .5); d [delay] -> D1 and D2 tie at 1, true D3 scores 0 and ties last: R@1 0
    assert abs(res["real_none_removed"]["R@1"][0] - 0.25) < 1e-9


def test_patient_view_removes_the_patients_own_paper_from_the_shortlist_and_resumes(tmp_path):
    ann, _ = kb(tmp_path)
    onto, sc = ann.onto, make_scorer("resnik_best_match", ann)
    kbp = PatientKB(ann, ["PMID:555"])
    assert kbp.view().overrides == {"OMIM:4": frozenset()} and kbp.diseases == ann.diseases
    cs = build_cases(REAL, ann)
    b = [c for c in cs if c["case_id"] == "PMID_555_b"]
    srv, url = serve()
    Stub.requests.clear()
    out = tmp_path / "real.jsonl"
    st = run_real(ann, sc, onto, b + [c for c in cs if c["case_id"] == "PMID_999_c"], out, endpoint=url, k=3, order="score", evidence="matches", min_ic=0.2)
    recs = [json.loads(l) for l in open(out)]
    assert st["new_records"] == 2 and len(recs) == 2 and not (tmp_path / "real.jsonl.one").exists()
    rb = [r for r in recs if r["case_id"] == "PMID_555_b"][0]
    assert "OMIM:4" not in [c["disease_id"] for c in rb["candidates"]] and rb["source"] == "real_all_removed"      # 4 diseases, K=3: D4 has nothing left, so it is not offered
    sent = json.dumps([q["messages"] for q in Stub.requests])
    assert "PMID" not in sent and "OMIM" not in sent
    again = run_real(ann, sc, onto, b, out, endpoint=url, k=3)
    assert again["new_records"] == 0 and again["skipped_done"] == 1 and len(Stub.requests) == 2
    srv.shutdown()
