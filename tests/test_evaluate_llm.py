from src.evaluate_llm import metrics, rank_in, summarize, versus


def rec(case, source, disease, cands, ranking, condition="rag", bad=0):
    return {"case_id": case, "source": source, "disease_id": disease, "condition": condition,
            "candidates": [{"disease_id": c, "score": 0.0} for c in cands], "ranking": ranking, "unmatched_lines": bad}


def test_rank_and_metrics():
    assert rank_in([["A"], ["B", "C"]], "C") == 2 and rank_in([["A"]], "Z") is None
    assert metrics(1) == {"R@1": 1.0, "R@10": 1.0, "MRR": 1.0} and metrics(None) == {"R@1": 0.0, "R@10": 0.0, "MRR": 0.0}
    assert metrics(11)["R@10"] == 0.0 and abs(metrics(4)["MRR"] - 0.25) < 1e-9


def test_summary_by_hand():
    recs = [rec("a", "s", "X1", ["X2", "X1", "X3"], [["X1"], ["X2"]]),        # model rank 1, scorer rank 2
            rec("b", "s", "X1", ["X1", "X2"], [["X2"], ["X1"]]),              # model rank 2, scorer rank 1
            rec("c", "s", "X2", ["X3", "X1"], [["X2"]])]                      # shortlist missed X2, the model named it anyway
    r = summarize(recs, n_boot=50)["s"]
    assert r["patients"] == 3 and r["diseases"] == 2 and r["offered"] == 2
    assert abs(r["llm"]["R@1"][0] - 0.75) < 1e-9 and abs(r["scorer"]["R@1"][0] - 0.25) < 1e-9 and abs(r["diff"]["R@1"][0] - 0.5) < 1e-9
    assert abs(r["llm"]["MRR"][0] - 0.875) < 1e-9 and abs(r["scorer"]["MRR"][0] - 0.375) < 1e-9
    assert "diff_offered" not in r                                            # fewer than five diseases: no interval on the subset


def test_model_alone_reports_the_unmatched_share_and_versus_pairs_runs():
    alone = [rec("a", "s", "X1", [], [["X1", "X9"]], condition="model_alone", bad=1), rec("b", "s", "X2", [], [["X5"]], condition="model_alone", bad=2)]
    r = summarize(alone, 20)["s"]
    assert abs(r["unmatched_share"] - 3 / 5) < 1e-9 and r["llm"]["R@1"][0] == 0.5 and "scorer" not in r     # 3 unmatched of 5 lines
    rag = [rec("a", "s", "X1", ["X1"], [["X1"]]), rec("b", "s", "X2", ["X3"], [["X3"]]), rec("z", "s", "X7", ["X7"], [["X7"]])]
    v = versus(rag, alone, 20)["s"]
    assert v["patients"] == 2 and abs(v["diff"]["R@1"][0] - 0.0) < 1e-9 and abs(v["diff"]["MRR"][0] - 0.0) < 1e-9    # shared a, b: rag 1/0, alone 1/0
