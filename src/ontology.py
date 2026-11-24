"""Human Phenotype Ontology reader (hp.obo).

Terms form a directed acyclic graph (is_a edges). This class loads it and answers the questions the rest of the
project asks:

  normalize(id)       the current ID for an ID that may be an alternative ID or an obsolete one with a replacement
                      (None if it cannot be resolved; failures are counted in .dropped)
  ancestors(t)        every term above t (not t itself)
  with_ancestors(t)   t plus every term above it
  descendants(t)      every term below t
  is_phenotypic(t)    True if t is, or is below, 'Phenotypic abnormality' (HP:0000118): inheritance, onset and so on are not
  name(t), synonyms(t)

    python -m src.ontology hp.obo
"""
import re
import sys
from collections import Counter, defaultdict

PHENOTYPIC_ROOT = "HP:0000118"
_ID = re.compile(r"HP:\d+")
_SYN = re.compile(r'^"(.*)"\s+(EXACT|BROAD|NARROW|RELATED)\b')


class Ontology:
    def __init__(self, path):
        self.names = {}
        self.parents = defaultdict(set)
        self.alt = {}
        self.replaced = {}
        self.obsolete = set()
        self._syn = defaultdict(list)
        self.version = None
        self.dropped = Counter()
        self._parse(path)
        self.children = defaultdict(set)
        for child, ps in self.parents.items():
            for p in ps:
                self.children[p].add(child)
        self._anc, self._desc = {}, {}

    def _parse(self, path):
        cur, in_term = None, False
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if line.startswith("["):
                    in_term = line == "[Term]"
                    cur = None
                    continue
                if not line or ":" not in line:
                    continue
                tag, _, val = line.partition(":")
                val = val.strip()
                if not in_term:
                    if tag == "data-version":
                        self.version = val
                    continue
                if tag == "id":
                    cur = val if val.startswith("HP:") else None
                elif cur is None:
                    continue
                elif tag == "name":
                    self.names[cur] = val
                elif tag == "is_a":
                    m = _ID.match(val)
                    if m:
                        self.parents[cur].add(m.group())
                elif tag == "alt_id":
                    self.alt[val.split()[0]] = cur
                elif tag == "replaced_by":
                    m = _ID.match(val)
                    if m:
                        self.replaced[cur] = m.group()
                elif tag == "is_obsolete" and val.lower() == "true":
                    self.obsolete.add(cur)
                elif tag == "synonym":
                    m = _SYN.match(val)
                    if m:
                        self._syn[cur].append((m.group(1), m.group(2)))

    # ---- identifiers ----
    def normalize(self, t):
        t = (t or "").strip()
        for _ in range(10):                                   # follow alt_id / replaced_by chains
            if t in self.alt:
                t = self.alt[t]
            elif t in self.obsolete and t in self.replaced:
                t = self.replaced[t]
            else:
                break
        if t in self.obsolete:
            self.dropped["obsolete, no replacement"] += 1
            return None
        if t not in self.names:
            self.dropped["unknown id"] += 1
            return None
        return t

    # ---- structure ----
    def ancestors(self, t):
        if t not in self._anc:
            out, stack = set(), list(self.parents.get(t, ()))
            while stack:
                p = stack.pop()
                if p not in out:
                    out.add(p)
                    stack.extend(self.parents.get(p, ()))
            self._anc[t] = frozenset(out)
        return self._anc[t]

    def with_ancestors(self, t):
        return self.ancestors(t) | {t}

    def descendants(self, t):
        if t not in self._desc:
            out, stack = set(), list(self.children.get(t, ()))
            while stack:
                c = stack.pop()
                if c not in out:
                    out.add(c)
                    stack.extend(self.children.get(c, ()))
            self._desc[t] = frozenset(out)
        return self._desc[t]

    def is_phenotypic(self, t):
        return t == PHENOTYPIC_ROOT or PHENOTYPIC_ROOT in self.ancestors(t)

    # ---- labels ----
    def name(self, t):
        return self.names.get(t, t)

    def synonyms(self, t, kinds=("EXACT",)):
        return [s for s, k in self._syn.get(t, ()) if k in kinds]

    def stats(self):
        live = [t for t in self.names if t not in self.obsolete]
        return {"version": self.version, "terms": len(live), "phenotypic terms": sum(1 for t in live if self.is_phenotypic(t)),
                "obsolete": len(self.obsolete), "alt ids": len(self.alt)}


if __name__ == "__main__":
    onto = Ontology(sys.argv[1] if len(sys.argv) > 1 else "hp.obo")
    for k, v in onto.stats().items():
        print(f"{k}: {v}")
