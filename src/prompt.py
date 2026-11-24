"""Prompts, the output grammar, and reply parsing for the LLM stage.

  build_rag_prompt    the patient's findings (names only, no IDs) and a shortlist of candidate diseases (names only, shuffled
                      deterministically so position carries no signal); the model must rank the candidates
  build_alone_prompt  the findings only; the model names the most likely diseases freely (the no-retrieval baseline)
  build_grammar       a GBNF grammar that forces the reply to be 1..K candidate names, one per line
  parse_ranking / parse_free_text   replies back into disease IDs
  check_no_leak       refuses a prompt that contains anything identifying the simulated patient or its profile
"""
import random

from src.names import strip_bullet

SYSTEM_RAG = (
    "You are a clinical decision support assistant for rare disease diagnosis.\n"
    "You will be given a patient's findings, coded as standardized phenotype terms, and a list of candidate diseases retrieved from a phenotype database.\n"
    "Rank the candidate diseases from most to least likely to explain the patient's findings.\n"
    "Rules:\n"
    "- Use only the candidates provided, copied exactly as written.\n"
    "- Write one disease per line, most likely first.\n"
    "- Write nothing else.")
SYSTEM_ALONE = (
    "You are a clinical decision support assistant for rare disease diagnosis.\n"
    "You will be given a patient's findings, coded as standardized phenotype terms.\n"
    "Name the diseases most likely to explain the findings, from most to least likely, using the standard disease name.\n"
    "Rules:\n"
    "- Write one disease per line, most likely first.\n"
    "- Write nothing else.")


SYSTEM_RAG_ORDERED = (
    "You are a clinical decision support assistant for rare disease diagnosis.\n"
    "You will be given a patient's findings, coded as standardized phenotype terms, and a list of candidate diseases ranked by a phenotype-matching program, best match first.\n"
    "Re-rank the candidates from most to least likely to explain the patient's findings. Keep the program's order unless the findings clearly support a different one.\n"
    "Rules:\n"
    "- Use only the candidates provided, copied exactly as written.\n"
    "- Write one disease per line, most likely first.\n"
    "- Write nothing else.")


def findings_text(present, onto):
    return "Findings present:\n" + "\n".join(f"- {n}" for n in sorted(onto.name(t) for t in present))


def unique_names(candidates):
    """Display names; a name shared by several diseases gets ' (2)', ' (3)' in disease-ID order."""
    seen, out = {}, {}
    for c in sorted(candidates, key=lambda c: c["disease_id"]):
        n = seen.get(c["name"], 0) + 1
        seen[c["name"]] = n
        out[c["disease_id"]] = c["name"] if n == 1 else f"{c['name']} ({n})"
    return out


def build_rag_prompt(present, onto, candidates, seed=0, order="shuffled", evidence="names"):
    """order 'shuffled': deterministic shuffle, no position signal. order 'score': the scorer's order, best first.
    evidence 'matches': under each candidate, which of the patient's findings match it (candidate dict needs 'matched_names')."""
    disp = unique_names(candidates)
    if order == "score":
        ids = [c["disease_id"] for c in candidates]
    else:
        ids = sorted(disp)
        random.Random(f"{seed}|" + ",".join(ids)).shuffle(ids)
    names = [disp[i] for i in ids]
    by_id = {c["disease_id"]: c for c in candidates}

    def block(i):
        line = f"- {disp[i]}"
        if evidence == "matches":
            m = by_id[i]["matched_names"]
            line += f"\n    matches {len(m)} of {len(present)} findings" + (": " + "; ".join(m) if m else "")
        return line

    user = findings_text(present, onto) + "\n\nCandidate diseases:\n" + "\n".join(block(i) for i in ids)
    system = SYSTEM_RAG_ORDERED if order == "score" else SYSTEM_RAG
    return {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "candidate_names": names, "name_to_id": {disp[i]: i for i in ids}}


def build_alone_prompt(present, onto, n_answers=10):
    user = findings_text(present, onto) + f"\n\nGive up to {n_answers} diseases."
    return {"messages": [{"role": "system", "content": SYSTEM_ALONE}, {"role": "user", "content": user}], "n_answers": n_answers}


def _q(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def build_grammar(names, max_items):
    m = max(1, min(max_items, len(names)))
    lines = ["root ::= item" + (" r1" if m > 1 else "")]
    for i in range(1, m):
        lines.append(f"r{i} ::= (nl item" + (f" r{i + 1}" if i + 1 < m else "") + ")?")
    lines += ["item ::= " + " | ".join(_q(n) for n in names), 'nl ::= "\\n"']
    return "\n".join(lines)


def parse_ranking(reply, name_to_id):
    """-> (ranking as a list of one-ID lists, number of lines that matched no candidate). Repeats are dropped."""
    lower = {n.casefold(): i for n, i in name_to_id.items()}
    out, seen, bad = [], set(), 0
    for line in reply.splitlines():
        line = strip_bullet(line)
        if not line:
            continue
        d = name_to_id.get(line) or lower.get(line.casefold())
        if d is None:
            bad += 1
        elif d not in seen:
            seen.add(d)
            out.append([d])
    return out, bad


def parse_free_text(reply, index):
    """-> (ranking as a list of ID lists, one list per matched line; unmatched lines). A name shared by several diseases keeps them all."""
    out, bad = [], 0
    for line in reply.splitlines():
        line = strip_bullet(line)
        if not line:
            continue
        ids = sorted(index.match(line))
        if ids:
            out.append(ids)
        else:
            bad += 1
    return out, bad


def check_no_leak(messages, case):
    """Raise if the prompt contains the case ID, profile ID, disease ID, a source paper, or marks the patient as simulated."""
    text = "\n".join(m["content"] for m in messages)
    forbidden = [case["case_id"], case["profile_id"], case["disease_id"], "SIM_", "PMID", "SIMULATED"] + list(case["exclude_pmids"])
    for f in forbidden:
        if f and f in text:
            raise ValueError(f"prompt for {case['case_id']} contains {f!r}")
