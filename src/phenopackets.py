"""Read Phenopacket Store files (GA4GH Phenopacket schema v2) into one record per patient.

Per patient: the HPO terms present and the ones recorded as excluded (both normalised through the ontology), sex,
the OMIM disease, the source paper(s) (PMID), the gene(s) and the cohort (the folder the file sits in). Patients with
no usable finding or no OMIM disease are dropped, and the reason is counted.

    python -m src.phenopackets --phenopackets phenopackets --hpo hp.obo --out cache/cases.jsonl
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


def _symbols(obj):
    """Gene symbols found under any 'geneContext' key."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "geneContext" and isinstance(v, dict) and v.get("symbol"):
                out.append(v["symbol"])
            else:
                out += _symbols(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _symbols(v)
    return out


def _diseases(d):
    """(id, label) pairs: the diagnosis in interpretations first, then the diseases list."""
    found = []
    for it in d.get("interpretations") or []:
        dis = ((it.get("diagnosis") or {}).get("disease")) or {}
        if dis.get("id"):
            found.append((dis["id"], dis.get("label", "")))
    for x in d.get("diseases") or []:
        term = x.get("term") or {}
        if term.get("id"):
            found.append((term["id"], term.get("label", "")))
    return found


def parse_phenopacket(d, onto, cohort):
    """-> (record, None) or (None, reason)."""
    pid = d.get("id", "")
    present, absent, remapped = set(), set(), 0
    for f in d.get("phenotypicFeatures") or []:
        raw = ((f.get("type") or {}).get("id")) or ""
        t = onto.normalize(raw)
        if t is None or not onto.is_phenotypic(t):
            continue
        remapped += t != raw.strip()
        (absent if f.get("excluded") else present).add(t)
    absent -= present
    if not present:
        return None, "no findings"
    omim = [(i, lab) for i, lab in _diseases(d) if i.startswith("OMIM:")]
    if not omim:
        return None, "no OMIM disease"
    pmids = sorted({r["id"] for r in ((d.get("metaData") or {}).get("externalReferences") or []) if str(r.get("id", "")).startswith("PMID:")})
    if not pmids:
        pmids = sorted({"PMID:" + m for m in re.findall(r"PMID_(\d+)", pid)})
    return {"case_id": pid, "cohort": cohort, "disease_id": omim[0][0], "disease_label": omim[0][1],
            "present": sorted(present), "absent": sorted(absent), "sex": (d.get("subject") or {}).get("sex", "UNKNOWN_SEX"),
            "pmids": pmids, "genes": sorted(set(_symbols(d))), "n_remapped": remapped}, None


def load_all(root, onto):
    root = Path(root)
    records, dropped, files = [], Counter(), 0
    for p in sorted(root.rglob("*.json")):
        files += 1
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            dropped["unreadable file"] += 1
            continue
        if not isinstance(d, dict) or "phenotypicFeatures" not in d:
            dropped["not a phenopacket"] += 1
            continue
        rel = p.relative_to(root).parts
        rec, why = parse_phenopacket(d, onto, rel[0] if len(rel) > 1 else "")
        if rec is None:
            dropped[why] += 1
        else:
            records.append(rec)
    per = Counter(r["disease_id"] for r in records)
    report = {"files": files, "cases": len(records), "dropped": dict(dropped),
              "terms_remapped": sum(r["n_remapped"] for r in records), "terms_dropped": dict(onto.dropped),
              "diseases": len(per), "cohorts": len({r["cohort"] for r in records}),
              "with_gene": sum(1 for r in records if r["genes"]), "with_exactly_one_pmid": sum(1 for r in records if len(r["pmids"]) == 1),
              "diseases_with_1_case": sum(1 for n in per.values() if n == 1), "diseases_with_10_or_more": sum(1 for n in per.values() if n >= 10)}
    return records, report


def main():
    from src.ontology import Ontology
    ap = argparse.ArgumentParser()
    ap.add_argument("--phenopackets", default="phenopackets")
    ap.add_argument("--hpo", default="hp.obo")
    ap.add_argument("--out", default="cache/cases.jsonl")
    a = ap.parse_args()
    records, report = load_all(a.phenopackets, Ontology(a.hpo))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for r in records:
            f.write(json.dumps({k: v for k, v in r.items() if k != "n_remapped"}) + "\n")
    json.dump(report, open(Path(a.out).parent / "load_report.json", "w"), indent=1)
    for k, v in report.items():
        print(f"{k}: {v}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
