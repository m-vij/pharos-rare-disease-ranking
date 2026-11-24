"""Check the parsed Phenomizer tables against the current ontology and annotation file.

For every syndrome: does its OMIM ID exist among the candidate diseases, and how many of its features have an
HPO ID that resolves in today's hp.obo (old IDs are followed through alt_id / replaced_by). Printed IDs that do not
resolve are listed so they can be fixed by hand.

    python -m src.validate_phenomizer data/phenomizer_tables.json
"""
import argparse
import json


def validate(tables, onto, disease_ids):
    out = []
    for t in tables:
        usable, unresolved = 0, []
        for f in t["features"]:
            ok = []
            for i in f["ids"]:
                norm = onto.normalize(i)
                if norm is not None and onto.is_phenotypic(norm):
                    ok.append(norm)
                else:
                    unresolved.append((f["name"], i))
            usable += bool(ok)
        out.append({"table": t["table"], "name": t["name"], "omim": t["omim"],
                    "in_annotations": (t["omim"] in disease_ids) if t["omim"] else None,
                    "features": len(t["features"]), "usable": usable, "unresolved": unresolved})
    return out


def main():
    from src.annotations import Annotations
    from src.ontology import Ontology
    ap = argparse.ArgumentParser()
    ap.add_argument("tables")
    ap.add_argument("--hpo", default="hp.obo")
    ap.add_argument("--hpoa", default="phenotype.hpoa")
    a = ap.parse_args()
    onto = Ontology(a.hpo)
    ann = Annotations(a.hpoa, onto)
    rows = validate(json.load(open(a.tables))["tables"], onto, set(ann.diseases))
    print(f"{'table':<6}{'syndrome':<46}{'OMIM':<13}{'in hpoa':<9}{'features':>9}{'usable':>8}")
    for r in rows:
        print(f"S{r['table']:<5}{r['name'][:44]:<46}{str(r['omim']):<13}{str(r['in_annotations']):<9}{r['features']:>9}{r['usable']:>8}")
    bad = [(r["table"], n, i) for r in rows for n, i in r["unresolved"]]
    print(f"\n{sum(r['usable'] for r in rows)} of {sum(r['features'] for r in rows)} features have a usable HPO term; "
          f"{sum(1 for r in rows if r['in_annotations'])} of {len(rows)} syndromes are in the annotation file")
    if bad:
        print("IDs that do not resolve (fix by hand):")
        for t, n, i in bad:
            print(f"  S{t}: {n[:55]!r}  {i}")


if __name__ == "__main__":
    main()
