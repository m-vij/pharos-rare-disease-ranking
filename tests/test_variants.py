import json

from src.fuse import fuse_summary, rrf_order
from src.prompt import SYSTEM_RAG, SYSTEM_RAG_ORDERED, build_rag_prompt
from src.run_llm import run_cases
from src.scorer import make_scorer
from tests.test_run_llm import Stub, cases, serve
from tests.test_scorer import kb


def test_explain_marks_a_finding_matched_only_when_the_shared_term_is_rare_enough(tmp_path):
    ann, v = kb(tmp_path)
    sc = make_scorer("resnik_best_match", ann)
    spasm_delay = "HP:0000013"                                              # D4 holds it itself (rarity ln 4 = 1.386); the others share only ln(4/3)
    assert sc.explain([spasm_delay], v, "OMIM:4", min_ic=1.0) == ([spasm_delay], [])
    assert sc.explain([spasm_delay], v, "OMIM:4", min_ic=2.0) == ([], [spasm_delay])
    assert sc.explain([spasm_delay], v, "OMIM:1", min_ic=1.0) == ([], [spasm_delay])
    assert sc.explain([spasm_delay], v, "OMIM:1", min_ic=0.2) == ([spasm_delay], [])


def test_prompt_variants(tmp_path):
    ann, _ = kb(tmp_path)
    cands = [{"disease_id": "OMIM:3", "name": "Zeta", "matched_names": ["Epileptic spasm"]},
             {"disease_id": "OMIM:1", "name": "Alpha", "matched_names": []},
             {"disease_id": "OMIM:2", "name": "Mid", "matched_names": ["Epileptic spasm", "Delay"]}]
    present = ["HP:0000011", "HP:0000012"]
    base = build_rag_prompt(present, ann.onto, cands, seed=1)
    assert base["messages"][0]["content"] == SYSTEM_RAG and "matches" not in base["messages"][1]["content"]
    ordered = build_rag_prompt(present, ann.onto, cands, seed=1, order="score", evidence="matches")
    assert ordered["candidate_names"] == ["Zeta", "Alpha", "Mid"] and ordered["messages"][0]["content"] == SYSTEM_RAG_ORDERED
    body = ordered["messages"][1]["content"]
    assert "- Zeta\n    matches 1 of 2 findings: Epileptic spasm\n- Alpha\n    matches 0 of 2 findings\n- Mid\n    matches 2 of 2 findings: Epileptic spasm; Delay" in body
    shuffled = build_rag_prompt(present, ann.onto, cands, seed=1, evidence="matches")
    assert sorted(shuffled["candidate_names"]) == ["Alpha", "Mid", "Zeta"] and shuffled["name_to_id"] == ordered["name_to_id"]


def test_run_with_evidence_and_the_scorers_order_reaches_the_server_without_leaks(tmp_path):
    ann, _ = kb(tmp_path)
    sc = make_scorer("resnik_best_match", ann)
    srv, url = serve()
    Stub.requests.clear()
    out = tmp_path / "v.jsonl"
    run_cases(ann, sc, ann.onto, cases(), out, condition="rag", k=3, endpoint=url, log_every=0, order="score", evidence="matches", min_ic=0.2)
    recs = [json.loads(l) for l in open(out)]
    assert recs[0]["order"] == "score" and recs[0]["evidence"] == "matches" and recs[0]["unmatched_lines"] == 0
    sent = json.dumps([q["messages"] for q in Stub.requests])
    assert "best match first" in sent and "matches " in sent and "SIM_" not in sent and "OMIM" not in sent and "PMID" not in sent
    srv.shutdown()
    refused = False
    try:
        run_cases(ann, make_scorer("exact_overlap", ann), ann.onto, cases(), tmp_path / "x.jsonl", condition="rag", k=3, endpoint="http://none", evidence="matches")
    except ValueError:
        refused = True
    assert refused


def test_fusion_by_hand():
    assert rrf_order(["A", "B", "C"], ["C", "A"], 60) == ["A", "C", "B"]                  # A: 1/61+1/62, C: 1/63+1/61, B: 1/62
    assert rrf_order(["A", "B"], ["B", "A"], 60) == ["A", "B"]                            # exact tie: the scorer's order wins
    rec = lambda cid, d, cands, ranking: {"case_id": cid, "source": "s", "disease_id": d, "condition": "rag",
                                          "candidates": [{"disease_id": c, "score": 0} for c in cands], "ranking": [[x] for x in ranking]}
    recs = [rec("a", "X1", ["X1", "X2", "X3"], ["X2", "X1", "X3"]), rec("b", "X2", ["X1", "X2", "X3"], ["X2", "X1", "X3"])]
    r = fuse_summary(recs, 60, 20)["s"]
    assert r["patients"] == 2 and r["diseases"] == 2
    assert abs(r["means"]["scorer"]["MRR"][0] - 0.75) < 1e-9 and abs(r["means"]["model"]["MRR"][0] - 0.75) < 1e-9
    assert abs(r["means"]["fused"]["MRR"][0] - 0.75) < 1e-9          # a: tie goes to the scorer (rank 1); b: X1 first, X2 second (rank 2)
