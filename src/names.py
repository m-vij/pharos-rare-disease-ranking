"""Exact disease-name matching (for the model-alone condition, where the model writes names freely).

Names are compared after light normalisation: case, punctuation, anything in parentheses, and 'type II' -> 'type 2'.
There is no fuzzy matching and no synonym list, so a correct answer written in different words does NOT match; the share of
unmatched lines is reported so the baseline's under-count is visible.
"""
import re
from collections import defaultdict

ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5", "vi": "6", "vii": "7", "viii": "8", "ix": "9", "x": "10"}
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


def strip_bullet(line):
    return _BULLET.sub("", line).strip()


def normalize(name):
    s = re.sub(r"\([^)]*\)", " ", name.casefold())
    toks = re.sub(r"[^a-z0-9]+", " ", s).split()
    return " ".join(ROMAN.get(t, t) if i and toks[i - 1] == "type" else t for i, t in enumerate(toks))


class NameIndex:
    def __init__(self, names):
        self.by = defaultdict(set)
        for d, n in names.items():
            self.by[normalize(n)].add(d)

    def match(self, name):
        return set(self.by.get(normalize(name), ()))
