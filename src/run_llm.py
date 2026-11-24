"""The LLM stage: for each simulated patient, build a prompt, ask the local model, parse the reply, save one JSON line.

  condition rag          the scorer picks the top-K candidates from the FULL annotation file; the model ranks them
                         (replies are forced by a grammar to be candidate names)
  condition model_alone  no candidates; the model names diseases freely; names are matched exactly to the disease list

The patient's identity never reaches the prompt (check_no_leak). Runs are resumable: cases already in the output file are
skipped, so use a new --tag whenever a setting changes. --dry-run builds and checks every prompt but sends nothing.

    python -m src.run_llm --endpoint http://127.0.0.1:8123 --per-profile 3
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.names import NameIndex
from src.prompt import build_alone_prompt, build_grammar, build_rag_prompt, check_no_leak, parse_free_text, parse_ranking


def chat(endpoint, messages, grammar, max_tokens, seed, timeout=600, retries=3):
    body = {"messages": messages, "temperature": 0, "seed": seed, "max_tokens": max_tokens}
    if grammar:
        body["grammar"] = grammar
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                out = json.load(r)
            msg = out["choices"][0]["message"]
            return msg.get("content") or "", bool(msg.get("reasoning_content")), out.get("usage", {})
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError) as e:
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"server request failed after {retries} tries: {last}")


def run_cases(ann, scorer, onto, cases, out_path, *, condition="rag", k=20, n_answers=10, seed=0, endpoint=None,
              max_tokens_per_item=40, timeout=600, log_every=50, dry_run=False, order="shuffled", evidence="names", min_ic=2.0):
    out_path = Path(out_path)
    done = set()
    if out_path.exists() and not dry_run:
        done = {json.loads(l)["case_id"] for l in open(out_path)}
    full = ann.view(())
    names = ann.names
    index = NameIndex({d: names[d] for d in ann.diseases if d in names}) if condition == "model_alone" else None
    tie = np.random.default_rng(seed).random(len(ann.diseases))
    f = None if dry_run else open(out_path, "a")
    stats = {"prompts": 0, "chars": [], "first_prompt": None, "skipped_done": 0, "new_records": 0}
    t0 = time.time()
    try:
        for c in cases:
            if c["case_id"] in done:
                stats["skipped_done"] += 1
                continue
            cands, grammar = [], None
            if condition == "rag":
                s = scorer.score(c["present"], full)
                top = np.lexsort((tie, -s))[:k]
                cands = [{"disease_id": ann.diseases[i], "name": names.get(ann.diseases[i], ann.diseases[i]), "score": float(s[i])} for i in top]
                if evidence == "matches":
                    if not hasattr(scorer, "explain"):
                        raise ValueError("--evidence matches needs a scorer with explain (resnik_best_match)")
                    for x in cands:
                        x["matched_names"] = sorted(onto.name(t) for t in scorer.explain(c["present"], full, x["disease_id"], min_ic)[0])
                p = build_rag_prompt(c["present"], onto, cands, seed, order=order, evidence=evidence)
                grammar, max_items = build_grammar(p["candidate_names"], len(cands)), len(cands)
            else:
                p = build_alone_prompt(c["present"], onto, n_answers)
                max_items = n_answers
            check_no_leak(p["messages"], c)
            stats["prompts"] += 1
            stats["chars"].append(sum(len(m["content"]) for m in p["messages"]))
            if stats["first_prompt"] is None:
                stats["first_prompt"] = p["messages"]
            if dry_run:
                continue
            t1 = time.time()
            reply, reasoning, usage = chat(endpoint, p["messages"], grammar, min(2048, max_tokens_per_item * max_items + 16), seed, timeout)
            ranking, bad = parse_ranking(reply, p["name_to_id"]) if condition == "rag" else parse_free_text(reply, index)
            f.write(json.dumps({"case_id": c["case_id"], "profile_id": c["profile_id"], "disease_id": c["disease_id"], "source": c["source"],
                                "condition": condition, "scorer": getattr(scorer, "name", None) if condition == "rag" else None, "k": k if condition == "rag" else None,
                                "order": order if condition == "rag" else None, "evidence": evidence if condition == "rag" else None,
                                "n_findings": len(c["present"]), "candidates": [{"disease_id": x["disease_id"], "score": x["score"]} for x in cands],
                                "messages": p["messages"], "reply": reply, "ranking": ranking, "unmatched_lines": bad, "has_reasoning": reasoning,
                                "usage": usage, "seconds": round(time.time() - t1, 2)}) + "\n")
            f.flush()
            stats["new_records"] += 1
            if log_every and stats["new_records"] % log_every == 0:
                print(f"{stats['new_records']} cases done, {(time.time() - t0) / stats['new_records']:.1f}s each", flush=True)
    finally:
        if f:
            f.close()
    return stats


def main():
    from src.annotations import Annotations
    from src.ontology import Ontology
    from src.scorer import Rarity, make_scorer
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--cases", default="cache/sim_cases.jsonl")
    ap.add_argument("--hpo", default="hp.obo")
    ap.add_argument("--hpoa", default="phenotype.hpoa")
    ap.add_argument("--split", default="config/split.json")
    ap.add_argument("--which", default="tuning", choices=["tuning", "heldout"])
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--condition", default="rag", choices=["rag", "model_alone"])
    ap.add_argument("--scorer", default="resnik_best_match")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--n-answers", type=int, default=10)
    ap.add_argument("--order", default="shuffled", choices=["shuffled", "score"], help="score = show the scorer's order, best first")
    ap.add_argument("--evidence", default="names", choices=["names", "matches"], help="matches = list which findings match each candidate")
    ap.add_argument("--min-ic", type=float, default=2.0, help="rarity a shared term needs for a finding to count as matched")
    ap.add_argument("--sources", default="phenopackets,phenomizer")
    ap.add_argument("--per-profile", type=int, default=3, help="use the first N simulated patients of each profile")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.which == "heldout" and not a.final:
        raise SystemExit("the held-out diseases are for one final run: add --final when every setting is frozen")
    keep = set(json.load(open(a.split))[a.which])
    seen, cases = defaultdict(int), []
    for line in open(a.cases):
        c = json.loads(line)
        if c["disease_id"] in keep and c["source"] in a.sources.split(","):
            seen[c["profile_id"]] += 1
            if seen[c["profile_id"]] <= a.per_profile:
                cases.append(c)
    onto = Ontology(a.hpo)
    ann = Annotations(a.hpoa, onto)
    scorer = make_scorer(a.scorer, ann, Rarity(ann)) if a.condition == "rag" else None
    out = a.out or f"cache/llm_{a.which}_{a.condition}{('_' + a.tag) if a.tag else ''}.jsonl"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    st = run_cases(ann, scorer, onto, cases, out, condition=a.condition, k=a.k, n_answers=a.n_answers, seed=a.seed, endpoint=a.endpoint,
                   timeout=a.timeout, dry_run=a.dry_run, order=a.order, evidence=a.evidence, min_ic=a.min_ic)
    ch = sorted(st["chars"])
    print(f"cases selected: {len(cases)}   prompts built: {st['prompts']}" + (f"   size (chars/4 ~ tokens): median {ch[len(ch) // 2] // 4}, max {ch[-1] // 4}" if ch else ""))
    if a.dry_run:
        for m in st["first_prompt"] or []:
            print(f"\n[{m['role']}]\n{m['content']}")
        print("\n(dry run: nothing sent, nothing written)")
    else:
        print(f"new records: {st['new_records']}  already done: {st['skipped_done']}  wrote {out}")


if __name__ == "__main__":
    main()
