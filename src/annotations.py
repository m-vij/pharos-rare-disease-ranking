"""Disease knowledge base: which phenotype terms each disease is annotated with (phenotype.hpoa).

Policies:
  - aspect P only (phenotypic abnormality; inheritance and onset rows are skipped)
  - only diseases whose ID prefix is listed (default OMIM)
  - 'NOT' rows are kept separately and never count as positive annotations
  - every positive annotation keeps its references (PMID:..., OMIM:...), so a later step can remove the rows that
    came from particular papers
  - closure(d) = the disease's terms plus all their ancestors (phenotypic ones)
  - view(pmids) = the same knowledge base minus every annotation that cites any of those papers

    python -m src.annotations hp.obo phenotype.hpoa
"""
import sys
from collections import Counter, defaultdict

# header names differ between HPO releases; map every variant to one key
_COLS = {
    "id": {"database_id", "databaseid", "db"},
    "name": {"disease_name", "db_name", "dbname"},
    "qualifier": {"qualifier"},
    "hpo": {"hpo_id", "hpoid"},
    "ref": {"reference", "db_reference", "dbreference", "db_ref"},
    "aspect": {"aspect"},
}


def _colmap(header):
    norm = [h.strip().lstrip("#").lower() for h in header]
    out = {}
    for key, names in _COLS.items():
        for i, h in enumerate(norm):
            if h in names:
                out[key] = i
                break
    missing = [k for k in ("id", "hpo", "aspect") if k not in out]
    if missing:
        raise ValueError(f"cannot find columns {missing} in header: {header}")
    return out


class Annotations:
    def __init__(self, hpoa_path, onto, prefixes=("OMIM",)):
        self.onto = onto
        self.prefixes = tuple(prefixes)
        self.names = {}
        self.pos = defaultdict(set)          # disease -> {term}
        self.neg = defaultdict(set)          # disease -> {term}
        self.refs = defaultdict(set)         # (disease, term) -> {reference strings}
        self.report = Counter()
        self.version = None
        self._read(hpoa_path)
        self.diseases = sorted(d for d in self.pos if self.pos[d])
        self._closures = {}
        self._pmid_index = None

    def _read(self, path):
        cm = None
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if cm is None:
                    if line.startswith("#") and len(parts) < 5:          # comment lines before the header
                        if line.lower().startswith("#version:"):
                            self.version = line.split(":", 1)[1].strip()
                        continue
                    cm = _colmap(parts)
                    continue
                self.report["rows"] += 1
                if parts[cm["aspect"]].strip() != "P":
                    self.report["rows_not_aspect_P"] += 1
                    continue
                did = parts[cm["id"]].strip()
                if did.split(":")[0] not in self.prefixes:
                    self.report["rows_other_prefix"] += 1
                    continue
                tid = self.onto.normalize(parts[cm["hpo"]].strip())
                if tid is None:
                    self.report["rows_unresolvable_term"] += 1
                    continue
                if not self.onto.is_phenotypic(tid):
                    self.report["rows_non_phenotypic"] += 1
                    continue
                if "name" in cm:
                    self.names[did] = parts[cm["name"]].strip()
                qual = parts[cm["qualifier"]].strip().upper() if "qualifier" in cm else ""
                if qual == "NOT":
                    self.neg[did].add(tid)
                    self.report["rows_NOT"] += 1
                else:
                    self.pos[did].add(tid)
                    self.report["rows_positive"] += 1
                    if "ref" in cm:
                        for r in parts[cm["ref"]].replace(",", ";").split(";"):
                            r = r.strip()
                            if r:
                                self.refs[(did, tid)].add(r)

    def closure(self, disease):
        """The disease's terms plus all their phenotypic ancestors."""
        if disease not in self._closures:
            out = set()
            for t in self.pos.get(disease, ()):
                out |= {a for a in self.onto.with_ancestors(t) if self.onto.is_phenotypic(a)}
            self._closures[disease] = frozenset(out)
        return self._closures[disease]

    def _by_pmid(self):
        if self._pmid_index is None:
            idx = defaultdict(set)
            for (d, t), refs in self.refs.items():
                for r in refs:
                    if r.startswith("PMID:"):
                        idx[r].add((d, t))
            self._pmid_index = idx
        return self._pmid_index

    def view(self, pmids=()):
        """The knowledge base as it would be without the given papers: every annotation citing any of them is removed."""
        return View(self, set(pmids))

    def stats(self):
        sizes = sorted(len(self.pos[d]) for d in self.diseases)
        return {"version": self.version, "candidate diseases": len(self.diseases),
                "annotations per disease (median)": sizes[len(sizes) // 2] if sizes else 0, **dict(self.report)}


class View:
    """Annotations without the rows that cite the given papers. Diseases untouched by the removal are read from the base."""

    def __init__(self, ann, pmids):
        self.ann = ann
        removed = defaultdict(set)
        idx = ann._by_pmid()
        for p in pmids:
            for d, t in idx.get(p, ()):
                removed[d].add(t)
        self.overrides = {d: frozenset(ann.pos[d] - ts) for d, ts in removed.items()}
        self._closures = {}

    def pos(self, d):
        return self.overrides[d] if d in self.overrides else frozenset(self.ann.pos.get(d, ()))

    def closure(self, d):
        if d not in self.overrides:
            return self.ann.closure(d)
        if d not in self._closures:
            out = set()
            for t in self.overrides[d]:
                out |= {a for a in self.ann.onto.with_ancestors(t) if self.ann.onto.is_phenotypic(a)}
            self._closures[d] = frozenset(out)
        return self._closures[d]

    def removed_fraction(self, d):
        base = len(self.ann.pos.get(d, ()))
        return 0.0 if not base else 1.0 - len(self.pos(d)) / base


if __name__ == "__main__":
    from src.ontology import Ontology
    onto = Ontology(sys.argv[1] if len(sys.argv) > 1 else "hp.obo")
    ann = Annotations(sys.argv[2] if len(sys.argv) > 2 else "phenotype.hpoa", onto)
    for k, v in ann.stats().items():
        print(f"{k}: {v}")
    print("terms dropped while reading (by reason):", dict(onto.dropped))
