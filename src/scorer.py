"""Scorers: rank every candidate disease for one patient's findings.

Each scorer takes the patient's findings and a knowledge-base VIEW (annotations without the papers a profile came from)
and returns one score per candidate disease; higher is better. The rarity of a term is its information content,
IC(t) = -ln(share of diseases whose closure contains t), computed once from the full annotation file.

  random              chance level (fixed by the patient, so reruns agree)
  annotation_count    how many terms the disease has; ignores the patient (shows what disease SIZE alone does)
  exact_overlap       how many of the patient's terms are directly annotated to the disease
  hierarchy_overlap   how many terms (the patient's own and all their ancestors, root excluded) the disease's closure also holds; no weights
  resnik_best_match   for each patient term, the rarity of the most informative term it shares with the disease; summed

Diseases whose annotations the view changed are rescored explicitly from the view; the rest use precomputed postings.
"""
import math
import zlib
from collections import Counter, defaultdict

import numpy as np

from src.ontology import PHENOTYPIC_ROOT

SCORERS = ("random", "annotation_count", "exact_overlap", "hierarchy_overlap", "resnik_best_match")


class Rarity:
    def __init__(self, ann):
        self.n = len(ann.diseases)
        self.counts = Counter()
        for d in ann.diseases:
            self.counts.update(ann.closure(d))

    def ic(self, t):
        c = self.counts.get(t, 0)
        return math.log(self.n / c) if c else math.log(self.n)


def _postings(pairs, idx):
    post = defaultdict(list)
    for d, terms in pairs:
        for t in terms:
            post[t].append(idx[d])
    return {t: np.array(v, dtype=np.int32) for t, v in post.items()}


class BaseScorer:
    name = "base"

    def __init__(self, ann):
        self.ann, self.onto = ann, ann.onto
        self.cand = ann.diseases
        self.idx = {d: i for i, d in enumerate(self.cand)}
        self.n = len(self.cand)
        self._anc = {}

    def _ancs(self, t):
        """The term and its phenotypic ancestors."""
        if t not in self._anc:
            self._anc[t] = tuple(sorted(a for a in self.onto.with_ancestors(t) if self.onto.is_phenotypic(a)))
        return self._anc[t]

    def _base(self, present):
        raise NotImplementedError

    def _one(self, present, pos_d, clos_d):
        raise NotImplementedError

    def score(self, present, view):
        present = sorted(set(present))
        s = self._base(present).astype(float)
        for d in view.overrides:
            if d in self.idx:
                s[self.idx[d]] = self._one(present, view.pos(d), view.closure(d))
        return np.round(s, 9)


class RandomScorer(BaseScorer):
    name = "random"

    def __init__(self, ann, seed=0):
        super().__init__(ann)
        self.seed = seed

    def score(self, present, view):
        key = zlib.crc32(",".join(sorted(set(present))).encode()) ^ self.seed
        return np.random.default_rng(key).random(self.n)


class AnnotationCount(BaseScorer):
    name = "annotation_count"

    def __init__(self, ann):
        super().__init__(ann)
        self.size = np.array([len(ann.pos[d]) for d in self.cand], dtype=float)

    def _base(self, present):
        return self.size.copy()

    def _one(self, present, pos_d, clos_d):
        return float(len(pos_d))


class ExactOverlap(BaseScorer):
    name = "exact_overlap"

    def __init__(self, ann):
        super().__init__(ann)
        self.post = _postings(((d, ann.pos[d]) for d in self.cand), self.idx)

    def _base(self, present):
        s = np.zeros(self.n)
        for t in present:
            p = self.post.get(t)
            if p is not None:
                s[p] += 1
        return s

    def _one(self, present, pos_d, clos_d):
        return float(len(set(present) & set(pos_d)))


class HierarchyOverlap(BaseScorer):
    name = "hierarchy_overlap"

    def __init__(self, ann):
        super().__init__(ann)
        self.post = _postings(((d, ann.closure(d)) for d in self.cand), self.idx)

    def _patient_closure(self, present):
        out = set()
        for t in present:
            out |= set(self._ancs(t))
        out.discard(PHENOTYPIC_ROOT)
        return out

    def _base(self, present):
        s = np.zeros(self.n)
        for c in self._patient_closure(present):
            p = self.post.get(c)
            if p is not None:
                s[p] += 1
        return s

    def _one(self, present, pos_d, clos_d):
        return float(len(self._patient_closure(present) & set(clos_d)))


class ResnikBestMatch(BaseScorer):
    name = "resnik_best_match"

    def __init__(self, ann, rarity):
        super().__init__(ann)
        self.rarity = rarity
        self.post = _postings(((d, ann.closure(d)) for d in self.cand), self.idx)

    def _base(self, present):
        s = np.zeros(self.n)
        for t in present:
            w = np.zeros(self.n)
            for c in self._ancs(t):
                p = self.post.get(c)
                if p is not None:
                    w[p] = np.maximum(w[p], self.rarity.ic(c))
            s += w
        return s

    def _one(self, present, pos_d, clos_d):
        return float(sum(max((self.rarity.ic(c) for c in self._ancs(t) if c in clos_d), default=0.0) for t in present))

    def explain(self, present, view, disease, min_ic=2.0):
        """-> (matched, unmatched) patient terms: matched if the most informative term shared with the disease has rarity >= min_ic."""
        clos = view.closure(disease)
        matched, unmatched = [], []
        for t in sorted(set(present)):
            best = max((self.rarity.ic(c) for c in self._ancs(t) if c in clos), default=0.0)
            (matched if best >= min_ic else unmatched).append(t)
        return matched, unmatched


def make_scorer(name, ann, rarity=None, seed=0):
    if name == "random":
        return RandomScorer(ann, seed)
    if name == "annotation_count":
        return AnnotationCount(ann)
    if name == "exact_overlap":
        return ExactOverlap(ann)
    if name == "hierarchy_overlap":
        return HierarchyOverlap(ann)
    if name == "resnik_best_match":
        return ResnikBestMatch(ann, rarity or Rarity(ann))
    raise ValueError(f"unknown scorer {name!r}; choose from {SCORERS}")
