from src.moves import moves, sign_test


def rec(case, source, disease, cands, first):
    return {"case_id": case, "source": source, "condition": "rag", "disease_id": disease,
            "candidates": [{"disease_id": c, "score": 0} for c in cands], "ranking": [[first]] if first else []}


def test_sign_test():
    assert abs(sign_test(1, 3) - 0.625) < 1e-9 and sign_test(0, 0) == 1.0 and sign_test(1, 1) == 1.0 and sign_test(5, 0) == 2 / 32


def test_moves_by_hand():
    recs = [rec("a", "s", "X1", ["X1", "X2"], "X1"),        # both right
            rec("b", "s", "X1", ["X1", "X2"], "X2"),        # damage
            rec("c", "s", "X1", ["X2", "X1"], "X1"),        # rescue
            rec("d", "s", "X1", ["X2", "X1"], "X2"),        # both wrong
            rec("e", "s", "X1", ["X2", "X3"], "X1"),        # shortlist missed the disease: not counted
            rec("f", "s", "X1", ["X1", "X2"], None)]        # empty reply counts as the model being wrong
    r = moves(recs)["s"]
    assert (r["both right"], r["damage"], r["rescue"], r["both wrong"]) == (1, 2, 1, 1) and abs(r["p"] - sign_test(2, 1)) < 1e-9
