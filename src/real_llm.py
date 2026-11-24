"""The LLM stage on REAL patients with leave-source-out scoring (secondary analysis; same prompts and settings as the frozen study).

For each real patient the shortlist, and the 'matches' evidence, come from the annotation file WITHOUT the paper that reported
the patient. Patients whose disease then has nothing left to match (real_all_removed) are skipped by default: the true disease
cannot be on any shortlist, so the model and the scorer would both score zero on them. The scorer-only table (src.real_eval)
still includes them. Output records use the stratum as `source`, so src.evaluate_llm reports each stratum separately.
No frozen file is touched: each patient is run through the frozen run_cases with a knowledge base fixed to that patient's view.

    python -m src.real_llm --endpoint http://127.0.0.1:8123 --per-disease 2 --order score --evidence matches --tag ordev
"""
import argparse
import json
from pathlib import Path

from src.real_eval import build_cases, load_real
from src.run_llm import run_cases


class PatientKB:
    """Looks like the annotation object run_cases expects, but view() always returns one patient's leave-source-out view."""

    def __init__(self, ann, pmids):
        self.ann, self.pmids = ann, pmids
        self.diseases, self.names = ann.diseases, ann.names

    def view(self, _=()):
        return self.ann.view(self.pmids)


def run_real(ann, scorer, onto, cases, out_path, *, endpoint=None, dry_run=False, **kw):
    out_path = Path(out_path)
    done = {json.loads(l)["case_id"] for l in open(out_path)} if out_path.exists() and not dry_run else set()
    one = out_path.with_name(out_path.name + ".one")
    new = skipped = prompts = 0
    first = None
    for c in cases:
        if c["case_id"] in done:
            skipped += 1
            continue
        if one.exists():
            one.unlink()
        st = run_cases(PatientKB(ann, c["exclude_pmids"]), scorer, onto, [c], one, condition="rag", endpoint=endpoint, log_every=0, dry_run=dry_run, **kw)
        prompts += st["prompts"]
        first = first or st["first_prompt"]
        if not dry_run:
            with open(one) as f, open(out_path, "a") as g:
                g.write(f.read())
            one.unlink()
            new += st["new_records"]
            if new % 50 == 0:
                print(f"{new} patients done", flush=True)
    return {"new_records": new, "skipped_done": skipped, "prompts": prompts, "first_prompt": first}


def main():
    from src.annotations import Annotations
    from src.ontology import Ontology
    from src.scorer import Rarity, make_scorer
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8080")
    ap.add_argument("--cases", default="cache/cases.jsonl")
    ap.add_argument("--hpo", default="hp.obo")
    ap.add_argument("--hpoa", default="phenotype.hpoa")
    ap.add_argument("--split", default="config/split.json")
    ap.add_argument("--include-tuning", action="store_true")
    ap.add_argument("--keep-all-removed", action="store_true", help="also run patients whose disease has nothing left to match")
    ap.add_argument("--per-disease", type=int, default=2)
    ap.add_argument("--scorer", default="resnik_best_match")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--order", default="shuffled", choices=["shuffled", "score"])
    ap.add_argument("--evidence", default="names", choices=["names", "matches"])
    ap.add_argument("--min-ic", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    exclude = set() if a.include_tuning else set(json.load(open(a.split))["tuning"])
    onto = Ontology(a.hpo)
    ann = Annotations(a.hpoa, onto)
    cases = build_cases(load_real(a.cases), ann, exclude, a.per_disease, skip_all_removed=not a.keep_all_removed)
    out = a.out or f"cache/llm_real_rag{('_' + a.tag) if a.tag else ''}.jsonl"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    st = run_real(ann, make_scorer(a.scorer, ann, Rarity(ann)), onto, cases, out, endpoint=a.endpoint, dry_run=a.dry_run, k=a.k, seed=a.seed,
                  timeout=a.timeout, order=a.order, evidence=a.evidence, min_ic=a.min_ic)
    by = {}
    for c in cases:
        by[c["source"]] = by.get(c["source"], 0) + 1
    print(f"patients selected: {len(cases)} {by}   prompts built: {st['prompts']}")
    if a.dry_run:
        for m in st["first_prompt"] or []:
            print(f"\n[{m['role']}]\n{m['content']}")
        print("\n(dry run: nothing sent, nothing written)")
    else:
        print(f"new records: {st['new_records']}  already done: {st['skipped_done']}  wrote {out}")


if __name__ == "__main__":
    main()
