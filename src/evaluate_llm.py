"""Score saved LLM runs (disease level), with 95% intervals from resampling diseases.

For each profile source: how the model ranked the true disease (R@1, R@10, MRR), for rag runs also the scorer's own order of the
SAME shortlist, the paired difference (model minus scorer) and, among patients whose shortlist contained the true disease, the
same difference. --versus compares two runs on the patients they share.

    python -m src.evaluate_llm cache/llm_tuning_rag.jsonl cache/llm_tuning_model_alone.jsonl
    python -m src.evaluate_llm cache/llm_tuning_rag.jsonl --versus cache/llm_tuning_model_alone.jsonl
"""
import argparse
import json
from collections import defaultdict

import numpy as np

from src.evaluate import bootstrap

KEYS = ("R@1", "R@10", "MRR")


def rank_in(ranking, true):
    for i, ids in enumerate(ranking, 1):
        if true in ids:
            return i
    return None


def metrics(rank):
    return {"R@1": float(rank == 1), "R@10": float(rank is not None and rank <= 10), "MRR": 1.0 / rank if rank else 0.0}


def disease_means(rows):
    """rows: (source, disease, metrics) -> {(source, disease): {metric: mean over that disease's patients}}"""
    g = defaultdict(list)
    for s, d, m in rows:
        g[(s, d)].append(m)
    return {k: {mk: float(np.mean([m[mk] for m in ms])) for mk in KEYS} for k, ms in g.items()}


def _ci(per_a, per_b, source, n_boot):
    ds = sorted(d for (s, d) in per_a if s == source and (source, d) in per_b)
    return {k: bootstrap([per_a[(source, d)][k] - per_b[(source, d)][k] for d in ds], n_boot) for k in KEYS} if ds else {}


def summarize(records, n_boot=2000):
    out = {}
    for src in sorted({r["source"] for r in records}):
        rs = [r for r in records if r["source"] == src]
        llm = [(src, r["disease_id"], metrics(rank_in(r["ranking"], r["disease_id"]))) for r in rs]
        pl = disease_means(llm)
        row = {"patients": len(rs), "diseases": len({r["disease_id"] for r in rs}),
               "llm": {k: bootstrap([v[k] for v in pl.values()], n_boot) for k in KEYS},
               "unmatched_share": None}
        lines = sum(len(r["ranking"]) + r["unmatched_lines"] for r in rs)
        if rs and rs[0]["condition"] == "model_alone" and lines:
            row["unmatched_share"] = sum(r["unmatched_lines"] for r in rs) / lines
        if rs and rs[0]["condition"] == "rag":
            ids = lambda r: [[c["disease_id"]] for c in r["candidates"]]
            sc = [(src, r["disease_id"], metrics(rank_in(ids(r), r["disease_id"]))) for r in rs]
            ps = disease_means(sc)
            off = [r for r in rs if rank_in(ids(r), r["disease_id"])]
            row.update({"offered": len(off), "scorer": {k: bootstrap([v[k] for v in ps.values()], n_boot) for k in KEYS},
                        "diff": _ci(pl, ps, src, n_boot)})
            if len({r["disease_id"] for r in off}) >= 5:
                o = {r["case_id"] for r in off}
                pl2 = disease_means([x for x, r in zip(llm, rs) if r["case_id"] in o])
                ps2 = disease_means([x for x, r in zip(sc, rs) if r["case_id"] in o])
                row["diff_offered"] = _ci(pl2, ps2, src, n_boot)
        out[src] = row
    return out


def versus(rec_a, rec_b, n_boot=2000):
    """Paired difference A minus B in the model's ranking, on the patients both runs contain."""
    shared = {r["case_id"] for r in rec_a} & {r["case_id"] for r in rec_b}
    out = {}
    for src in sorted({r["source"] for r in rec_a}):
        pa = disease_means([(src, r["disease_id"], metrics(rank_in(r["ranking"], r["disease_id"]))) for r in rec_a if r["case_id"] in shared and r["source"] == src])
        pb = disease_means([(src, r["disease_id"], metrics(rank_in(r["ranking"], r["disease_id"]))) for r in rec_b if r["case_id"] in shared and r["source"] == src])
        out[src] = {"patients": sum(1 for r in rec_a if r["case_id"] in shared and r["source"] == src), "diff": _ci(pa, pb, src, n_boot)}
    return out


def _fmt(t):
    return f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"


def print_summary(path, res):
    print(f"\n{path}")
    for src, r in res.items():
        extra = f"   shortlist held the true disease for {r['offered']} patients" if "offered" in r else ""
        extra += f"   unmatched answer lines: {r['unmatched_share']:.0%}" if r["unmatched_share"] is not None else ""
        print(f"  source {src}: {r['diseases']} diseases, {r['patients']} patients{extra}")
        for name, block in (("model", r["llm"]), ("scorer's own order, same shortlist", r.get("scorer")),
                            ("model minus scorer, all patients", r.get("diff")), ("model minus scorer, shortlist held the disease", r.get("diff_offered"))):
            if block:
                print(f"    {name:<46}" + "   ".join(f"{k} {_fmt(block[k])}" for k in KEYS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--versus")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    load = lambda p: [json.loads(l) for l in open(p)]
    for p in a.runs:
        recs = load(p)
        if recs:
            print_summary(p, summarize(recs, a.boot))
        if a.versus and recs:
            print(f"  {p} MINUS {a.versus} (same patients, model ranking):")
            for src, r in versus(recs, load(a.versus), a.boot).items():
                print(f"    {src} ({r['patients']} patients): " + "   ".join(f"{k} {_fmt(r['diff'][k])}" for k in KEYS if r["diff"]))


if __name__ == "__main__":
    main()
