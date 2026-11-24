"""Rescue and damage: among patients whose shortlist held the true disease, who put it FIRST?

  both right      scorer and model both ranked it first
  damage          the scorer had it first and the model moved it down
  rescue          the scorer did not have it first and the model did
  both wrong      neither put it first
The sign test compares damage with rescue. It treats patients as independent although several come from the same disease, so its
p-value is optimistic; read it as a direction, not a precise probability.

    python -m src.moves cache/llm_tuning_rag.jsonl cache/llm_tuning_rag_ordev.jsonl
"""
import argparse
import json
from math import comb


def sign_test(a, b):
    n = a + b
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n)


def moves(records):
    out = {}
    for src in sorted({r["source"] for r in records}):
        both = damage = rescue = neither = 0
        for r in records:
            if r["source"] != src or r["condition"] != "rag":
                continue
            cand = [c["disease_id"] for c in r["candidates"]]
            if r["disease_id"] not in cand:
                continue
            s_ok = cand[0] == r["disease_id"]
            m_ok = bool(r["ranking"]) and r["disease_id"] in r["ranking"][0]
            both += s_ok and m_ok
            damage += s_ok and not m_ok
            rescue += (not s_ok) and m_ok
            neither += (not s_ok) and not m_ok
        out[src] = {"both right": both, "damage": damage, "rescue": rescue, "both wrong": neither, "p": sign_test(damage, rescue)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    for p in a.runs:
        print(p)
        for src, r in moves([json.loads(l) for l in open(p)]).items():
            print(f"  {src}: both right {r['both right']} | damage {r['damage']} | rescue {r['rescue']} | both wrong {r['both wrong']} | sign test p = {r['p']:.3g}")


if __name__ == "__main__":
    main()
