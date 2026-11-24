import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from src.run_llm import run_cases
from src.scorer import make_scorer
from tests.test_scorer import kb


class Stub(BaseHTTPRequestHandler):
    requests = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.requests.append(body)
        user = body["messages"][-1]["content"]
        if "Candidate diseases:" in user:
            names = [l[2:] for l in user.split("Candidate diseases:")[1].splitlines() if l.startswith("- ")]
            reply = "\n".join(names[:2])
        else:
            reply = "D1\nSome unknown disease"
        data = json.dumps({"choices": [{"message": {"content": reply}}], "usage": {"prompt_tokens": 1}}).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def serve():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Stub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def cases():
    return [{"case_id": "SIM_OMIM1_phenopackets_0", "profile_id": "OMIM:1|phenopackets", "disease_id": "OMIM:1", "source": "phenopackets",
             "present": ["HP:0000011"], "exclude_pmids": ["PMID:111"]},
            {"case_id": "SIM_OMIM4_phenomizer_0", "profile_id": "OMIM:4|phenomizer", "disease_id": "OMIM:4", "source": "phenomizer",
             "present": ["HP:0000013"], "exclude_pmids": []}]


def test_rag_run_with_a_stub_server_resume_and_dry_run(tmp_path):
    ann, _ = kb(tmp_path)
    onto, sc = ann.onto, make_scorer("resnik_best_match", ann)
    srv, url = serve()
    Stub.requests.clear()
    out = tmp_path / "run.jsonl"
    st = run_cases(ann, sc, onto, cases(), out, condition="rag", k=3, endpoint=url, log_every=0)
    recs = [json.loads(l) for l in open(out)]
    assert st["new_records"] == 2 and len(recs) == 2
    r = recs[0]
    assert len(r["candidates"]) == 3 and len(r["ranking"]) == 2 and all(len(x) == 1 for x in r["ranking"]) and r["unmatched_lines"] == 0
    assert r["condition"] == "rag" and r["scorer"] == "resnik_best_match" and r["k"] == 3 and r["disease_id"] == "OMIM:1"
    req = Stub.requests[0]
    assert req["temperature"] == 0 and "seed" in req and req["grammar"].startswith("root ::= item r1")
    sent = json.dumps([q["messages"] for q in Stub.requests])
    assert "SIM_" not in sent and "OMIM" not in sent and "PMID" not in sent and "phenopackets" not in sent
    again = run_cases(ann, sc, onto, cases(), out, condition="rag", k=3, endpoint=url, log_every=0)
    assert again["new_records"] == 0 and again["skipped_done"] == 2 and len(Stub.requests) == 2
    dry = run_cases(ann, sc, onto, cases(), tmp_path / "dry.jsonl", condition="rag", k=3, dry_run=True)
    assert dry["prompts"] == 2 and not (tmp_path / "dry.jsonl").exists()
    srv.shutdown()


def test_model_alone_run_matches_names_exactly_and_sends_no_grammar(tmp_path):
    ann, _ = kb(tmp_path)
    srv, url = serve()
    Stub.requests.clear()
    out = tmp_path / "alone.jsonl"
    run_cases(ann, None, ann.onto, cases(), out, condition="model_alone", endpoint=url, log_every=0)
    r = json.loads(open(out).readline())
    assert "grammar" not in Stub.requests[0] and r["candidates"] == [] and r["scorer"] is None
    assert r["ranking"] == [["OMIM:1"]] and r["unmatched_lines"] == 1                      # 'D1' matched exactly; the invented name did not
    srv.shutdown()
