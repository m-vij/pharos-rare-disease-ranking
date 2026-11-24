"""Fuse the scorer's order with the model's ranking (reciprocal rank fusion), from saved ranking runs. No GPU.

Each disease gets 1/(k + rank) from each ordering and the two are added; k = 60 is the usual constant and is NOT tuned. Ties go to
the scorer's order. Fusion can only reorder the shortlist, so it cannot raise the share of patients whose shortlist holds the disease.

    python -m src.fuse cache/llm_tuning_rag.jsonl
"""
import argparse
import json

from src.evaluate import bootstrap
from src.evaluate_llm import KEYS, disease_means, metrics, rank_in


def rrf_order(scorer_ids, llm_ids, k=60):
    pts = {}
    for order in (scorer_ids, llm_ids):
        for r, d in enumerate(order, 1):
            pts[d] = pts.get(d, 0.0) + 1.0 / (k + r)
    pos = {d: i for i, d in enumerate(scorer_ids)}
    return sorted(pts, key=lambda d: (-pts[d], pos.get(d, len(pos))))


def fuse_summary(records, k=60, n_boot=2000):
    out = {}
    for src in sorted({r["source"] for r in records}):
        rs = [r for r in records if r["source"] == src and r["condition"] == "rag"]
        rows = {n: [] for n in ("scorer", "model", "fused")}
        for r in rs:
            s_ids = [c["disease_id"] for c in r["candidates"]]
            m_ids = [x[0] for x in r["ranking"]]
            for name, ids in (("scorer", s_ids), ("model", m_ids), ("fused", rrf_order(s_ids, m_ids, k))):
                rows[name].append((src, r["disease_id"], metrics(rank_in([[d] for d in ids], r["disease_id"]))))
        per = {n: disease_means(v) for n, v in rows.items()}
        ds = sorted(per["fused"])
        out[src] = {"patients": len(rs), "diseases": len(ds),
                    "means": {n: {m: bootstrap([per[n][d][m] for d in ds], n_boot) for m in KEYS} for n in per},
                    "fused minus scorer": {m: bootstrap([per["fused"][d][m] - per["scorer"][d][m] for d in ds], n_boot) for m in KEYS},
                    "fused minus model": {m: bootstrap([per["fused"][d][m] - per["model"][d][m] for d in ds], n_boot) for m in KEYS}}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    fmt = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    for p in a.runs:
        recs = [json.loads(l) for l in open(p)]
        print(f"\n{p}   (95% intervals over diseases)")
        for src, r in fuse_summary(recs, a.k, a.boot).items():
            print(f"  source {src}: {r['diseases']} diseases, {r['patients']} patients")
            for n in ("scorer", "model", "fused"):
                print(f"    {n:<20}" + "   ".join(f"{m} {fmt(r['means'][n][m])}" for m in KEYS))
            for n in ("fused minus scorer", "fused minus model"):
                print(f"    {n:<20}" + "   ".join(f"{m} {fmt(r[n][m])}" for m in KEYS))


if __name__ == "__main__":
    main()
