"""Journal-style figures with one consistent color theme (blue = language model, vermillion = scorer, grey = baselines and data). Every plotted number is read from the saved result files; none is typed in here.

    python figures/make_figures.py           # writes figures/figure1 ... figure5 (pdf and png)
    python figures/make_figures.py --check   # prints the parsed numbers and stops
Inputs: results/{annotations,final_scorers,real_scorers,final_llm_primary,final_llm_ablation,real_llm,final_moves,final_moves_ablation}.txt,
        cache/load_report.json, cache/profiles.json, config/split.json, data/phenomizer_tables.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"], "font.size": 8,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
NUM = r"(-?\d+\.\d+) \[(-?\d+\.\d+), (-?\d+\.\d+)\]"
# One color per role, used in every figure (colorblind-safe Okabe-Ito shades; they also separate in grayscale by marker and line style).
C_MODEL, C_SCORER, C_BASE, C_PURPLE, C_GOLD = "#0072B2", "#D55E00", "#7F7F7F", "#CC79A7", "#E69F00"


def parse_scorers(text):
    out, src = {}, None
    for line in text.splitlines():
        m = re.match(r"^source: (\w+)\s+diseases: (\d+)\s+patients: (\d+)", line)
        if m:
            src = m.group(1)
            out[src] = {"diseases": int(m.group(2)), "patients": int(m.group(3))}
            continue
        m = re.match(rf"^(\w+)\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s*$", line)
        if m and src:
            out[src][m.group(1)] = {k: tuple(float(x) for x in m.groups()[1 + 3 * i: 4 + 3 * i]) for i, k in enumerate(("R@1", "R@10", "R@20", "MRR"))}
    return out


def parse_llm(text):
    out, run, src = {}, None, None
    for line in text.splitlines():
        m = re.match(r"^\S*?(llm_\w+)\.jsonl$", line.strip())
        if m:
            run, src = m.group(1), None
            out[run] = {}
            continue
        m = re.match(r"^\s+source (\w+): (\d+) diseases, (\d+) patients", line)
        if m and run:
            src = m.group(1)
            out[run][src] = {"diseases": int(m.group(2)), "patients": int(m.group(3))}
            continue
        m = re.match(rf"^\s+(model|scorer's own order, same shortlist|model minus scorer, all patients)\s+R@1 {NUM}\s+R@10 {NUM}\s+MRR {NUM}", line)
        if m and run and src:
            key = {"model": "model", "model minus scorer, all patients": "diff"}.get(m.group(1), "scorer")
            out[run][src][key] = {k: tuple(float(x) for x in m.groups()[1 + 3 * i: 4 + 3 * i]) for i, k in enumerate(("R@1", "R@10", "MRR"))}
    return out


def parse_moves(text):
    out, run = {}, None
    for line in text.splitlines():
        m = re.match(r"^\S*?(llm_\w+)\.jsonl$", line.strip())
        if m:
            run = m.group(1)
            out[run] = {}
            continue
        m = re.match(r"^\s+(\w+): both right (\d+) \| damage (\d+) \| rescue (\d+) \| both wrong (\d+)", line)
        if m and run:
            out[run][m.group(1)] = dict(zip(("both", "damage", "rescue", "wrong"), (int(x) for x in m.groups()[1:])))
    return out


def read(root, rel, how):
    p = Path(root) / rel
    if not p.exists():
        sys.exit(f"missing {p}   (create it with: {how})")
    return p.read_text()


def collect(root):
    d = {"held": parse_scorers(read(root, "results/final_scorers.txt", "python -m src.evaluate --which heldout --final | tee results/final_scorers.txt")),
         "real": parse_scorers(read(root, "results/real_scorers.txt", "python -m src.real_eval | tee results/real_scorers.txt"))}
    d["llm"] = parse_llm(read(root, "results/final_llm_primary.txt", "python -m src.evaluate_llm cache/llm_heldout_rag_ordev.jsonl cache/llm_heldout_model_alone.jsonl | tee results/final_llm_primary.txt"))
    d["llm"].update(parse_llm(read(root, "results/final_llm_ablation.txt", "python -m src.evaluate_llm cache/llm_heldout_rag.jsonl cache/llm_heldout_rag_ev.jsonl cache/llm_heldout_rag_ord.jsonl | tee results/final_llm_ablation.txt")))
    d["llm"].update(parse_llm(read(root, "results/real_llm.txt", "python -m src.evaluate_llm cache/llm_real_rag_ordev.jsonl | tee results/real_llm.txt")))
    d["moves"] = parse_moves(read(root, "results/final_moves.txt", "python -m src.moves cache/llm_heldout_rag_ordev.jsonl | tee results/final_moves.txt"))
    d["moves"].update(parse_moves(read(root, "results/final_moves_ablation.txt", "python -m src.moves cache/llm_heldout_rag.jsonl cache/llm_heldout_rag_ev.jsonl cache/llm_heldout_rag_ord.jsonl | tee results/final_moves_ablation.txt")))
    m = re.search(r"candidate diseases: (\d+)", read(root, "results/annotations.txt", "python -m src.annotations hp.obo phenotype.hpoa | tee results/annotations.txt"))
    lr = json.loads(read(root, "cache/load_report.json", "python -m src.phenopackets"))
    prof = json.loads(read(root, "cache/profiles.json", "python -m src.profiles"))
    split = json.loads(read(root, "config/split.json", "python -m src.split"))
    tables = json.loads(read(root, "data/phenomizer_tables.json", "copy the file to data/"))["tables"]
    d["n"] = {"kb": int(m.group(1)), "real": lr["cases"], "real_dis": lr["diseases"], "p_cohort": sum(p["source"] == "phenopackets" for p in prof),
              "p_syn": sum(p["source"] == "phenomizer" for p in prof), "syn": len(tables), "feat": sum(len(t["features"]) for t in tables),
              "tune": len(split["tuning"]), "held": len(split["heldout"])}
    return d


def save(fig, out, name):
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def figure1(d, out):
    n = d["n"]
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, ec=C_BASE):
        ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.5, linespacing=1.3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="black", shrinkA=0, shrinkB=0))

    box(0.02, 0.84, 0.28, 0.12, f"Phenomizer supplement\n{n['syn']} syndromes\n{n['feat']:,} features")
    box(0.36, 0.84, 0.28, 0.12, f"Phenopacket Store\n{n['real']:,} patients\n{n['real_dis']} diseases")
    box(0.70, 0.84, 0.28, 0.12, f"HPO annotation file\n{n['kb']:,} OMIM diseases")
    box(0.02, 0.60, 0.50, 0.15, f"Disease profiles\n{n['p_cohort']} from patient cohorts\n(at least 10 patients)\n{n['p_syn']} from syndrome tables")  # layout-fixed
    box(0.02, 0.42, 0.50, 0.12, "Simulated patients\n30 per profile")
    box(0.56, 0.42, 0.28, 0.12, "Real cases (own source\npaper removed from\nthe annotations)")
    box(0.02, 0.22, 0.96, 0.12, f"Scorers rank all {n['kb']:,} diseases\nTop 20 form the shortlist", ec=C_SCORER)
    box(0.02, 0.02, 0.46, 0.12, "Language model re-ranks\nthe shortlist (4 prompt designs)", ec=C_MODEL)
    box(0.52, 0.02, 0.46, 0.12, f"Evaluation by disease\n{n['tune']} tuning / {n['held']} held-out diseases", ec="black")
    for a in [(0.16, 0.84, 0.16, 0.75), (0.44, 0.84, 0.44, 0.75), (0.58, 0.84, 0.66, 0.54), (0.27, 0.60, 0.27, 0.54), (0.27, 0.42, 0.27, 0.34),
              (0.70, 0.42, 0.70, 0.34), (0.91, 0.84, 0.91, 0.34), (0.25, 0.22, 0.25, 0.14), (0.75, 0.22, 0.75, 0.14), (0.48, 0.08, 0.52, 0.08)]:
        arrow(*a)
    save(fig, out, "figure1")


def dots(ax, rows, markers, colors=None):
    ys = list(range(len(rows)))[::-1]
    colors = colors or ["black"] * len(rows)
    for y, (label, (v, lo, hi)), filled, col in zip(ys, rows, markers, colors):
        ax.errorbar(v, y, xerr=[[v - lo], [hi - v]], fmt="o", color=col, mfc=col if filled else "white", mec=col, ms=4.8, lw=0.9, capsize=2.5, capthick=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    return ys


def forest(ax, labels, blocks, xlim, title, letter):
    """Paired difference (model minus scorer's own order): filled circle = first-choice recall, open square = MRR; 95% interval."""
    n = len(labels)
    for i, blk in enumerate(blocks):
        y = n - 1 - i
        for off, key, marker, fill in ((0.15, "R@1", "o", C_MODEL), (-0.15, "MRR", "s", "white")):
            v, lo, hi = blk["diff"][key]
            ax.errorbar(v, y + off, xerr=[[v - lo], [hi - v]], fmt=marker, color=C_MODEL, mfc=fill, mec=C_MODEL, ms=4.8, lw=0.9, capsize=2.5, capthick=0.9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(n - 1, -1, -1))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(*xlim)
    ax.set_title(title, fontsize=8, loc="left")
    ax.text(-0.02, 1.16, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")


def figure2(d, out):
    """Recall at 1, 10 and 20 for the five scorers on the held-out simulated patients."""
    names = [("resnik_best_match", "Rarity-weighted", "-", "o", C_SCORER), ("exact_overlap", "Exact overlap", "--", "s", C_GOLD), ("hierarchy_overlap", "Hierarchy overlap", ":", "^", C_PURPLE),
             ("annotation_count", "Annotation count", "-", "x", C_BASE), ("random", "Random", "--", "+", C_BASE)]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, src, letter, title in ((axes[0], "phenopackets", "a", "Cohort-derived"), (axes[1], "phenomizer", "b", "Syndrome-derived")):
        blk = d["held"][src]
        for i, (key, label, ls, mk, col) in enumerate(names):
            xs = [0, 1, 2]
            vals = [blk[key][k] for k in ("R@1", "R@10", "R@20")]
            off = (i - 2) * 0.05
            ax.errorbar([x + off for x in xs], [v[0] for v in vals], yerr=[[v[0] - v[1] for v in vals], [v[2] - v[0] for v in vals]], color=col, ls=ls, marker=mk, ms=4,
                        lw=0.9, capsize=2, capthick=0.7, label=label)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["1", "10", "20"])
        ax.set_xlabel("Shortlist size (K)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{title} ({blk['diseases']} diseases)", fontsize=8, loc="left")
        ax.text(-0.02, 1.16, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
    axes[0].set_ylabel("Recall at K")
    axes[1].legend(loc="center right", frameon=False, fontsize=6.5)
    fig.tight_layout()
    save(fig, out, "figure2")


def figure3(d, out):
    """Absolute first-choice recall of the language model for each prompt design, against the scorer's own order of the same shortlist."""
    runs = [("Unaided model", "llm_heldout_model_alone"), ("Shuffled names", "llm_heldout_rag"), ("Shuffled names + matched findings", "llm_heldout_rag_ev"),
            ("Scorer's order", "llm_heldout_rag_ord"), ("Scorer's order + matched findings", "llm_heldout_rag_ordev")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7), sharey=True)
    for ax, src, letter, name in ((axes[0], "phenopackets", "a", "Cohort-derived"), (axes[1], "phenomizer", "b", "Syndrome-derived")):
        rows = [(lab, d["llm"][r][src]["model"]["R@1"]) for lab, r in runs]
        dots(ax, rows, [False, True, True, True, True], [C_BASE, C_MODEL, C_MODEL, C_MODEL, C_MODEL])
        ax.axvline(d["llm"]["llm_heldout_rag_ordev"][src]["scorer"]["R@1"][0], color=C_SCORER, lw=1.2, ls="--")
        ax.set_xlim(0, 1)
        ax.set_xlabel("First-choice recall")
        ax.set_title(f"{name} ({d['llm']['llm_heldout_rag_ordev'][src]['diseases']} diseases)", fontsize=8, loc="left")
        ax.text(-0.02, 1.16, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
    fig.legend([plt.Line2D([], [], color=C_SCORER, lw=1.2, ls="--"), plt.Line2D([], [], marker="o", ls="", color=C_MODEL, ms=4.8), plt.Line2D([], [], marker="o", ls="", color=C_BASE, mfc="white", ms=4.8)],
               ["Scorer's own order of the same shortlist", "Language model, shortlist given", "Language model, unaided"], ncol=3, loc="lower center", frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, out, "figure3")


def figure4(d, out):
    prompts = [("Shuffled names", "llm_heldout_rag"), ("Shuffled names + matched findings", "llm_heldout_rag_ev"), ("Scorer's order", "llm_heldout_rag_ord"),
               ("Scorer's order + matched findings", "llm_heldout_rag_ordev")]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.0), gridspec_kw={"width_ratios": [1.1, 1.0, 0.8], "wspace": 0.95})
    for ax, src, letter, name in ((axes[0], "phenopackets", "a", "Simulated, cohort-derived"), (axes[1], "phenomizer", "b", "Simulated, syndrome-derived")):
        forest(ax, [p[0] for p in prompts] if letter == "a" else [""] * 4, [d["llm"][r][src] for _, r in prompts], (-0.6, 0.2),
               f"{name}\n({d['llm']['llm_heldout_rag_ordev'][src]['diseases']} diseases)", letter)
    real = d["llm"]["llm_real_rag_ordev"]
    forest(axes[2], ["None removed", "Partly removed"], [real["real_none_removed"], real["real_partly_removed"]], (-0.12, 0.12), "Real cases\n(own paper removed)", "c")
    for ax in axes:
        ax.set_xlabel("Model minus scorer", fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=C_MODEL, ms=4.8), plt.Line2D([], [], marker="s", ls="", color=C_MODEL, mfc="white", ms=4.8)]
    fig.legend(handles, ["First-choice recall", "Mean reciprocal rank"], loc="lower center", ncol=2, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(left=0.20, right=0.985, top=0.78, bottom=0.27)
    save(fig, out, "figure4")


def figure5(d, out):
    """What the model did to the scorer's first choice, among held-out simulated patients whose shortlist held the true disease."""
    prompts = [("Shuffled names", "llm_heldout_rag"), ("Shuffled names + matched findings", "llm_heldout_rag_ev"), ("Scorer's order", "llm_heldout_rag_ord"),
               ("Scorer's order + matched findings", "llm_heldout_rag_ordev")]
    parts = [("both", "Both right", "#595959", None), ("damage", "Model overturned a correct scorer choice", C_SCORER, None),
             ("rescue", "Model corrected a scorer miss", C_MODEL, None), ("wrong", "Both wrong", "white", None)]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.8))
    for ax, src, letter, name in ((axes[0], "phenopackets", "a", "Cohort-derived"), (axes[1], "phenomizer", "b", "Syndrome-derived")):
        for i, (lab, run) in enumerate(prompts):
            c = d["moves"][run][src]
            tot = sum(c.values())
            left, y = 0.0, len(prompts) - 1 - i
            for key, _, col, _h in parts:
                w = c[key] / tot
                ax.barh(y, w, left=left, color=col, edgecolor="black", lw=0.6, height=0.62)
                if w >= 0.08:
                    ax.text(left + w / 2, y, str(c[key]), ha="center", va="center", fontsize=6.5, color="black" if col == "white" else "white")
                left += w
        ax.set_yticks(range(len(prompts) - 1, -1, -1))
        ax.set_yticklabels([p[0] for p in prompts] if letter == "a" else [""] * len(prompts))
        ax.set_ylim(-0.5, len(prompts) - 0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Share of patients\n(shortlist held the true disease)")
        ax.set_title(name, fontsize=8, loc="left")
        ax.text(-0.02, 1.16, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=col, ec="black", lw=0.6) for _, _, col, _h in parts]
    fig.legend(handles, [p[1] for p in parts], loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save(fig, out, "figure5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    d = collect(a.root)
    if a.check:
        print(d["n"])
        for run, blk in d["llm"].items():
            for s, b in blk.items():
                print(run, s, b["diseases"], "diseases", "model R@1", b["model"]["R@1"], "diff R@1", b["diff"]["R@1"] if "diff" in b else "-")
        for k in ("held", "real"):
            for s, b in d[k].items():
                print(k, s, b["diseases"], "diseases", b["patients"], "patients", b["resnik_best_match"]["R@10"])
        return
    out = Path(a.root) / a.out
    out.mkdir(parents=True, exist_ok=True)
    figure1(d, out)
    figure2(d, out)
    figure3(d, out)
    figure4(d, out)
    figure5(d, out)
    print(f"wrote figure1 to figure5 (pdf and png) to {out}")


if __name__ == "__main__":
    main()
