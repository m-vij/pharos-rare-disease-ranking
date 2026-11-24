"""Supplementary workflow figures. Case-flow counts are read from the saved result files (same inputs as make_figures.py); the second figure is a
schematic with generic example labels and no data.

    python figures/make_supp_figures.py      # writes figures/supp_figure1 (case flow) and supp_figure2 (leave-source-out example), pdf and png
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import FancyArrowPatch, Rectangle, collect, plt, save  # noqa: E402


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, bw, bh, text, fs=7.5, fc="white", lw=1.2, ec="#7F7F7F"):
        ax.add_patch(Rectangle((x, y), bw, bh, fc=fc, ec=ec, lw=lw))
        ax.text(x + bw / 2, y + bh / 2, text, ha="center", va="center", fontsize=fs, linespacing=1.3)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="black", shrinkA=0, shrinkB=0))

    return fig, ax, box, arrow


def supp1(d, lr, out):
    n, h, r, llm = d["n"], d["held"], d["real"], d["llm"]
    dropped = sum(lr.get("dropped", {}).values())
    held_patients = sum(b["patients"] for b in h.values())
    sim_llm = sum(llm["llm_heldout_rag_ordev"][s]["patients"] for s in ("phenopackets", "phenomizer"))
    real_llm = sum(llm["llm_real_rag_ordev"][s]["patients"] for s in ("real_none_removed", "real_partly_removed"))
    fig, ax, box, arrow = canvas(7.0, 6.2)
    box(0.30, 0.90, 0.40, 0.08, f"Phenopacket files read\n{lr['files']:,}")
    box(0.74, 0.86, 0.25, 0.08, f"Excluded: {dropped}\n(no usable findings)", fs=7)
    box(0.30, 0.74, 0.40, 0.09, f"Real patients\n{n['real']:,} patients, {n['real_dis']} diseases")
    ax.text(0.262, 0.662, "Simulation", ha="left", fontsize=8, fontweight="bold")
    ax.text(0.772, 0.662, "Real-case analysis", ha="left", fontsize=8, fontweight="bold")
    box(0.02, 0.52, 0.45, 0.12, f"Disease profiles\n{n['p_cohort']} from patient cohorts\n(at least 10 patients)\n{n['p_syn']} from {n['syn']} syndromes")  # layout-fixed
    box(0.02, 0.36, 0.45, 0.12, f"Split by disease: {n['tune']} tuning, {n['held']} held-out\nHeld-out simulated patients scored: {held_patients:,}\n({h['phenopackets']['diseases']} cohort- and {h['phenomizer']['diseases']} syndrome-derived diseases)", fs=7)
    box(0.02, 0.17, 0.45, 0.13, f"Language model sample\n{sim_llm} held-out simulated patients\n(3 per profile)", ec="#0072B2")
    box(0.53, 0.54, 0.45, 0.10, f"Tuning diseases and diseases outside the\nannotation file excluded\n{r['real_all']['patients']:,} patients, {r['real_all']['diseases']} diseases")
    box(0.53, 0.33, 0.45, 0.15, f"Leave-source-out groups\nnone removed: {r['real_none_removed']['patients']:,}\npartly removed: {r['real_partly_removed']['patients']:,}\nall removed: {r['real_all_removed']['patients']:,} (cannot be ranked)", ec="#D55E00")
    box(0.53, 0.14, 0.45, 0.14, f"Language model sample\n{real_llm} patients, at most 2 per disease\n({llm['llm_real_rag_ordev']['real_none_removed']['patients']} none removed, {llm['llm_real_rag_ordev']['real_partly_removed']['patients']} partly removed)", ec="#0072B2")
    ax.plot([0.50, 0.50], [0.74, 0.69], color="black", lw=0.8)
    ax.plot([0.245, 0.755], [0.69, 0.69], color="black", lw=0.8)
    for a in [(0.50, 0.90, 0.50, 0.83), (0.70, 0.90, 0.74, 0.90), (0.245, 0.69, 0.245, 0.64), (0.755, 0.69, 0.755, 0.64), (0.245, 0.52, 0.245, 0.48), (0.245, 0.36, 0.245, 0.30),
              (0.755, 0.54, 0.755, 0.48), (0.755, 0.33, 0.755, 0.28)]:
        arrow(*a)
    save(fig, out, "supp_figure1")


def supp2(out):
    fig, ax, box, arrow = canvas(6.8, 3.7)
    cols = [("None removed", [("Finding 1", "Paper B"), ("Finding 2", "Paper B"), ("Finding 3", "Paper C"), ("Finding 4", "Paper C")], [], "Scored against\nall four annotations"),
            ("Partly removed", [("Finding 1", "Paper A"), ("Finding 2", "Papers A, B"), ("Finding 3", "Paper C"), ("Finding 4", "Paper C")], [0, 1], "Scored against the\ntwo that remain"),
            ("All removed", [("Finding 1", "Paper A"), ("Finding 2", "Paper A"), ("Finding 3", "Paper A"), ("Finding 4", "Paper A")], [0, 1, 2, 3], "Nothing left: the disease\ncannot be ranked")]
    ax.text(0.5, 0.98, "Patient reported in Paper A: annotations citing Paper A are removed from that disease before scoring", ha="center", va="top", fontsize=8)
    for i, (title, rows, gone, result) in enumerate(cols):
        x0 = 0.02 + i * 0.33
        ax.text(x0 + 0.15, 0.86, title, ha="center", fontsize=8, fontweight="bold")
        ax.text(x0 + 0.015, 0.78, "Annotation", fontsize=7)
        ax.text(x0 + 0.165, 0.78, "Cited in", fontsize=7)
        for j, (a, b) in enumerate(rows):
            y = 0.66 - j * 0.115
            removed = j in gone
            ax.add_patch(Rectangle((x0, y), 0.30, 0.095, fc="0.85" if removed else "white", ec="black", lw=0.7))
            ax.text(x0 + 0.015, y + 0.0475, a, va="center", fontsize=7, color="0.35" if removed else "black")
            ax.text(x0 + 0.165, y + 0.0475, b, va="center", fontsize=7, color="0.35" if removed else "black")
            if removed:
                ax.plot([x0 + 0.01, x0 + 0.29], [y + 0.0475, y + 0.0475], color="0.35", lw=0.8)
        box(x0, 0.04, 0.30, 0.12, result, fs=7, ec="#D55E00")
        arrow(x0 + 0.15, 0.20, x0 + 0.15, 0.165)
    save(fig, out, "supp_figure2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="figures")
    a = ap.parse_args()
    d = collect(a.root)
    lr = json.loads((Path(a.root) / "cache/load_report.json").read_text())
    out = Path(a.root) / a.out
    out.mkdir(parents=True, exist_ok=True)
    supp1(d, lr, out)
    supp2(out)
    print(f"wrote supp_figure1 and supp_figure2 (pdf and png) to {out}")


if __name__ == "__main__":
    main()
