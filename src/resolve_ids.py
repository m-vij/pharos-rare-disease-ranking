"""Repair the HPO IDs printed in the Phenomizer tables automatically, and say what was done.

The 2009 supplement has typos in some IDs (a digit too many or too few, a wrong prefix). For every printed ID:
  1. if it is a valid, current term (alternative and obsolete IDs are followed), it is used as printed;
  2. otherwise candidate repairs are generated (pad with zeros to 7 digits, drop an extra leading zero, add a missing last
     digit 0-9) and the candidate whose label or synonym best matches the feature's printed NAME is chosen, but only if the
     match is good (similarity >= MIN_SIM) and clearly better than the runner-up (by MARGIN); else the ID is dropped;
  3. a row with no printed ID is looked up by exact name against labels and exact synonyms (unique match only).
Every decision is returned in a log so nothing is silent.
"""
import difflib
import re

MIN_SIM = 0.6
MARGIN = 0.1


def _norm(s):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s.casefold()).split())


def name_parts(name):
    parts = [p.strip() for p in re.split(r"\s+(?:and/or|and|or)\s+", name, flags=re.I)]
    return [p for p in parts if p] or [name]


def resolve_quiet(onto, t):
    """Like onto.normalize, but without counting failures, and phenotypic terms only."""
    for _ in range(10):
        if t in onto.alt:
            t = onto.alt[t]
        elif t in onto.obsolete and t in onto.replaced:
            t = onto.replaced[t]
        else:
            break
    return t if (t in onto.names and t not in onto.obsolete and onto.is_phenotypic(t)) else None


def _repairs(printed):
    digits = printed.split(":", 1)[1] if ":" in printed else printed
    out = []
    if len(digits) < 7:
        out.append(("zero-padded", "HP:" + digits.zfill(7)))
        if len(digits) == 6:
            out += [("missing last digit", "HP:" + digits + str(k)) for k in range(10)]
    elif len(digits) > 7:
        stripped = digits.lstrip("0")
        if 0 < len(stripped) <= 7:
            out.append(("extra leading zero", "HP:" + stripped.zfill(7)))
    return out


class Resolver:
    def __init__(self, onto):
        self.onto = onto
        self._labels = {}
        self._index = None

    def labels(self, t):
        if t not in self._labels:
            self._labels[t] = [self.onto.name(t)] + self.onto.synonyms(t, ("EXACT", "RELATED"))
        return self._labels[t]

    def similarity(self, t, name):
        return max(difflib.SequenceMatcher(None, _norm(p), _norm(l)).ratio() for p in name_parts(name) for l in self.labels(t))

    def by_name(self):
        if self._index is None:
            idx = {}
            for t in self.onto.names:
                if t in self.onto.obsolete or not self.onto.is_phenotypic(t):
                    continue
                for lab in [self.onto.name(t)] + self.onto.synonyms(t, ("EXACT",)):
                    idx.setdefault(_norm(lab), set()).add(t)
            self._index = idx
        return self._index

    def resolve_feature(self, name, printed_ids):
        """-> (ids, log). log entries: {'printed', 'result', 'method', 'similarity'}."""
        ids, log = [], []

        def add(t):
            if t not in ids:
                ids.append(t)

        for pid in printed_ids:
            t = resolve_quiet(self.onto, pid)
            if t:
                add(t)
                log.append({"printed": pid, "result": t, "method": "as printed" if t == pid else "followed alt/replaced id", "similarity": None})
                continue
            cands = []
            for why, c in _repairs(pid):
                t2 = resolve_quiet(self.onto, c)
                if t2:
                    cands.append((self.similarity(t2, name), t2, why))
            cands.sort(reverse=True)
            if cands and cands[0][0] >= MIN_SIM and (len(cands) == 1 or cands[0][0] - cands[1][0] >= MARGIN):
                add(cands[0][1])
                log.append({"printed": pid, "result": cands[0][1], "method": "repaired: " + cands[0][2], "similarity": round(cands[0][0], 2)})
            else:
                best = f"best candidate {cands[0][1]} at {cands[0][0]:.2f}" if cands else "no candidate"
                log.append({"printed": pid, "result": None, "method": "dropped: " + best, "similarity": None})
        if not printed_ids:
            for part in name_parts(name):
                hit = self.by_name().get(_norm(part), set())
                if len(hit) == 1:
                    t = next(iter(hit))
                    add(t)
                    log.append({"printed": "", "result": t, "method": "matched by exact name", "similarity": 1.0})
                else:
                    log.append({"printed": "", "result": None, "method": f"dropped: name matches {len(hit)} terms", "similarity": None})
        return ids, log
