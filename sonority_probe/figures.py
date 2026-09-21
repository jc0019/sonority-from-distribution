"""The paper's figures, drawn from the analysis CSVs.

    python -m sonority_probe.figures [--latex-dir latex/figures]

Reads outputs/eval, outputs/probe, outputs/hw_trajectory and outputs/per_phoneme
(see the options) and writes the 16 figures to outputs/figures; --latex-dir
copies them into the paper's figure directory as well.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from sonority_probe.config import ARCH_LAYERS, CONDITIONS, TRAIN, TRAJECTORY_EPOCHS

ARCHS = tuple(ARCH_LAYERS)  # lstm, tf2, tf4
ARCH_TITLES = {"lstm": "LSTM", "tf2": "Transformer 2L", "tf4": "Transformer 4L"}
CONDITION_NAMES = {"feat": "Feat", "emb": "Emb", "perm": "Perm", "rand": "Rand"}
CONDITION_COLOURS = {"feat": "#009E73", "emb": "#0072B2", "perm": "#D55E00", "rand": "#CC79A7"}  # Okabe-Ito
SPACES = ("input", "hidden_bos", "hidden_in_context")
SPACE_TITLES = {"input": "Input geometry", "hidden_bos": "BOS-conditioned", "hidden_in_context": "In-context"}
PROBE_YLABEL = "LOOCV Spearman $\\rho$"
SURPRISAL_YLABEL = "Mean $\\Delta$surprisal\n(violator $-$ control)"
EPOCH_TICKS = (0, 5, 10, 15, 20, 25, 30, 35, 40)
PAPER_FIGURES = (
    "bnif_lbif_contours.pdf",
    "consonant_probe_decomposition_bestval.png",
    "cv_binary_probe_trajectory_notitle.png",
    "consonant_probe_trajectory_notitle.png",
    "tf2_consonants_trajectory_notitle.png",
    "contour_expectations_lstm.png",
    "contour_expectations_tf2.png",
    "contour_expectations_tf4.png",
    "hw_discrimination.png",
    "daland_vs_bigram.png",
    "per_phoneme_displacement.png",
    "rsa_retention_to_input.png",
    "trajectory_grid_full60.png",
    "trajectory_grid_cv_binary.png",
    "trajectory_grid_consonants.png",
    "hw_trajectory_grid.png",
)


def _save(fig, out_dir: Path, name: str) -> None:
    fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_dir / f"{name}.png")


def _zero_line(ax) -> None:
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.35)


def _top_legend(fig, ax, *, fontsize: float, ncol: int = 4) -> None:
    """One frameless legend centred above the panels."""
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False, fontsize=fontsize,
               bbox_to_anchor=(0.5, 1.0), columnspacing=1.8, handlelength=1.8)


def _epochs(checkpoints: pd.Series) -> pd.Series:
    return checkpoints.map(lambda c: 0 if c == "step_0" else int(c.split("_")[1]))


def _require_complete_trajectory(cell: pd.DataFrame, label: str) -> None:
    """A mean curve must average every seed at every checkpoint, once."""
    expected = {(seed, epoch) for seed in TRAIN.seeds for epoch in (0, *TRAJECTORY_EPOCHS)}
    actual = list(zip(cell["seed"], cell["epoch"]))
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f"{label}: expected every seed exactly once at every trajectory checkpoint")


# Fig. 1: schematic sonority contours of bnif and lbif (no data).

SCHEMATIC_WORDS = (  # word, phonemes, sonority ranks, colour, description
    ("bnif", ("b", "n", "i", "f"), (1, 2, 5, 1), "#0072B2", "rising onset: well-formed"),
    ("lbif", ("l", "b", "i", "f"), (3, 1, 5, 1), "#D55E00", "falling onset: ill-formed"),
)
SCHEMATIC_ROLES = ((0.5, "onset"), (2.0, "nucleus"), (3.0, "coda"))
SCHEMATIC_CLASSES = ("Obstruents", "Nasals", "Liquids", "Glides", "Vowels")


def plot_sonority_projection_example(out_dir: Path) -> None:
    # Drawn 7.2 in wide and scaled by LaTeX to (0.85 of) the 7.7 cm column, so the
    # text prints at (0.85 of) 8 pt. Saved as PDF to keep the text vector.
    scale = 7.2 / (7.7 / 2.54)
    body, small, title = 8 * scale, 7.5 * scale, 9.5 * scale
    rc = {
        "font.size": body, "axes.titlesize": title,
        "xtick.labelsize": body, "ytick.labelsize": body,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white",
    }
    rank_labels = [f"{name} ({rank})" for rank, name in enumerate(SCHEMATIC_CLASSES, start=1)]
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.5), sharey=True)
        for ax, (word, phones, ranks, colour, description) in zip(axes, SCHEMATIC_WORDS):
            ax.axvspan(-0.5, 1.5, color=colour, alpha=0.16, linewidth=0, zorder=0)  # the onset
            ax.plot(range(4), ranks, color=colour, linewidth=1.8, marker="o", markersize=3.2, zorder=3)
            ax.set_title(word, fontstyle="italic", pad=9.5 * scale)
            ax.text(0.5, 1.03, description, transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=small, color=colour)
            ax.set_xlim(-0.5, 3.5)
            ax.set_ylim(0.6, 5.4)
            ax.set_xticks(range(4), [f"/{p}/" for p in phones])
            ax.set_yticks(range(1, 6), rank_labels)
            ax.tick_params(axis="both", length=3, pad=2)
            ax.grid(True, alpha=0.25)
            ax.set_axisbelow(True)
            # Syllabic roles as a second tier of x labels below the phonemes.
            role_offset = -(3 + 2 + 1.15 * body + 0.15 * body)  # tick + pad + label height + gap, in points
            for x, role in SCHEMATIC_ROLES:
                is_onset = role == "onset"
                ax.annotate(role, xy=(x, 0), xycoords=ax.get_xaxis_transform(),
                            xytext=(0, role_offset), textcoords="offset points",
                            ha="center", va="top", fontsize=small,
                            color=colour if is_onset else "black",
                            fontweight="bold" if is_onset else "normal",
                            annotation_clip=False)
        fig.supylabel("Sonority rank", fontsize=body, x=0.012)
        fig.tight_layout(pad=0.4, h_pad=0.9)
        fig.savefig(out_dir / "bnif_lbif_contours.pdf", bbox_inches=None)  # keep the width exact
        plt.close(fig)
        print("wrote", out_dir / "bnif_lbif_contours.pdf")


# Fig. 2: probe decomposition at the best-validation checkpoint.

PROBE_BARS = (  # probe, legend label, colour
    ("full60", "Full60", "#0072B2"),
    ("cv_binary", "CV-binary", "#009E73"),
    ("consonants", "Consonants", "#E69F00"),
    ("shuffled_full60", "Shuffled (Full60)", "#999999"),
)


def plot_decomposition(summary: pd.DataFrame, out_dir: Path) -> None:
    FS = 16
    x = np.arange(len(SPACES))
    width = 0.2
    fig, axes = plt.subplots(3, 4, figsize=(16, 8.8), sharex=True, sharey=True)
    for row, arch in enumerate(ARCHS):
        best = summary[(summary.arch == arch) & (summary.checkpoint == "best_val")]
        for ax, condition in zip(axes[row], CONDITIONS):
            for i, (probe, label, colour) in enumerate(PROBE_BARS):
                cell = best[(best.condition == condition) & (best.probe == probe)].set_index("extraction").reindex(SPACES)
                means = cell["mean_rho"].to_numpy()
                yerr = np.vstack([means - cell["seed_min"].to_numpy(), cell["seed_max"].to_numpy() - means])
                yerr = np.clip(yerr, 0, None)  # identical seeds (Feat input) can put the mean 1 ulp outside the range
                ax.bar(x + (i - 1.5) * width, means, width, label=label, color=colour,
                       yerr=yerr, ecolor="#555", capsize=2, error_kw={"lw": 0.8})
            _zero_line(ax)
            if row == 0:
                ax.set_title(CONDITION_NAMES[condition], fontsize=FS + 3)
            ax.set_xticks(x, [SPACE_TITLES[space] for space in SPACES], rotation=20, ha="right", fontsize=FS)
            ax.tick_params(axis="y", labelsize=FS)
            ax.set_ylim(-0.5, 1.0)
            ax.grid(axis="y", alpha=0.25)
        axes[row, 0].set_ylabel(ARCH_TITLES[arch], fontsize=FS + 1)
    fig.supylabel(PROBE_YLABEL, fontsize=FS + 1, x=0.008)
    _top_legend(fig, axes[0, 0], fontsize=FS + 1)
    fig.tight_layout(rect=[0.02, 0, 1, 0.94], h_pad=1.2)
    _save(fig, out_dir, "consonant_probe_decomposition_bestval")


# Figs. 3, 4 and 12-14: probe scores across training.

TRAJECTORY_YLIM = (-0.6, 0.9)  # the same in every trajectory figure; covers every seed line
TRAJECTORY_YTICKS = (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75)


def _trajectory_lines(ax, cell: pd.DataFrame, value: str, condition: str, *, sem_band: bool = False) -> None:
    """Thin lines for the seeds, a thick line for their mean, optionally a +-1 SEM band."""
    colour = CONDITION_COLOURS[condition]
    for _, seed_rows in cell.groupby("seed"):
        seed_rows = seed_rows.sort_values("epoch")
        ax.plot(seed_rows["epoch"], seed_rows[value], color=colour, alpha=0.18, linewidth=0.75)
    agg = cell.groupby("epoch")[value].agg(["mean", "std", "size"]).sort_index()
    ax.plot(agg.index, agg["mean"], color=colour, linewidth=1.7, marker="o", markersize=2.6,
            label=CONDITION_NAMES[condition])
    if sem_band:
        sem = agg["std"] / np.sqrt(agg["size"])
        ax.fill_between(agg.index, agg["mean"] - sem, agg["mean"] + sem, color=colour, alpha=0.12, linewidth=0)


def plot_probe_trajectory(scores: pd.DataFrame, arch: str, probe: str, name: str, out_dir: Path,
                          *, sem_band: bool = False) -> None:
    """One architecture and probe: the three representation spaces side by side."""
    sub = scores[(scores.arch == arch) & (scores.probe == probe) & (scores.checkpoint != "best_val")].copy()
    sub["epoch"] = _epochs(sub["checkpoint"])
    FS = 12
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, space in zip(axes, SPACES):
        for condition in CONDITIONS:
            cell = sub[(sub.extraction == space) & (sub.condition == condition)]
            _require_complete_trajectory(cell, f"{name}/{space}/{condition}")
            _trajectory_lines(ax, cell, "spearman_rho", condition, sem_band=sem_band)
        _zero_line(ax)
        ax.set_title(SPACE_TITLES[space], fontsize=FS + 1)
        ax.set_xticks(EPOCH_TICKS)
        ax.tick_params(axis="both", labelsize=FS - 1)
        ax.set_xlabel("Epoch", fontsize=FS)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylim(*TRAJECTORY_YLIM)
    axes[0].set_yticks(TRAJECTORY_YTICKS)
    axes[0].set_ylabel(PROBE_YLABEL, fontsize=FS)
    _top_legend(fig, axes[0], fontsize=FS)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    _save(fig, out_dir, name)


def plot_probe_trajectory_grid(scores: pd.DataFrame, probe: str, name: str, out_dir: Path) -> None:
    """One probe: architectures in rows, representation spaces in columns."""
    sub = scores[(scores.probe == probe) & (scores.checkpoint != "best_val")].copy()
    sub["epoch"] = _epochs(sub["checkpoint"])
    FS = 14
    fig, axes = plt.subplots(len(ARCHS), len(SPACES), figsize=(13.5, 10.8), sharex=True, sharey=True)
    for r, arch in enumerate(ARCHS):
        for c, space in enumerate(SPACES):
            ax = axes[r, c]
            for condition in CONDITIONS:
                cell = sub[(sub.arch == arch) & (sub.extraction == space) & (sub.condition == condition)]
                _require_complete_trajectory(cell, f"{name}/{arch}/{space}/{condition}")
                _trajectory_lines(ax, cell, "spearman_rho", condition)
            _zero_line(ax)
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=FS - 1)
            if r == 0:
                ax.set_title(SPACE_TITLES[space], fontsize=FS + 1)
            if c == 0:
                ax.set_ylabel(f"{ARCH_TITLES[arch]}\n{PROBE_YLABEL}", fontsize=FS)
            if r == len(ARCHS) - 1:
                ax.set_xticks(EPOCH_TICKS)
                ax.set_xlabel("Epoch", fontsize=FS)
    axes[0, 0].set_ylim(*TRAJECTORY_YLIM)
    axes[0, 0].set_yticks(TRAJECTORY_YTICKS)
    _top_legend(fig, axes[0, 0], fontsize=FS + 1)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=1.0)
    _save(fig, out_dir, name)


# Fig. 9: similarity of each representation space to the input geometry (RSA).

def plot_rsa_retention(rsa_summary: pd.DataFrame, out_dir: Path) -> None:
    FS = 12
    fig, axes = plt.subplots(1, len(ARCHS), figsize=(4.0 * len(ARCHS), 3.6), sharey=True)
    for ax, arch in zip(axes, ARCHS):
        rows = rsa_summary[rsa_summary.arch == arch]
        for j, condition in enumerate(CONDITIONS):
            xs = [0, 1 + (j - 1.5) * 0.045, 2 + (j - 1.5) * 0.045]
            ys, lo, hi = [1.0], [0.0], [0.0]  # the input's similarity to itself is 1
            for pair in ("input<->hidden_bos", "input<->hidden_in_context"):
                r = rows[(rows.condition == condition) & (rows.pair == pair)].iloc[0]
                ys.append(r["mean_rho"])
                lo.append(r["mean_rho"] - r["seed_min"])
                hi.append(r["seed_max"] - r["mean_rho"])
            ax.errorbar(xs, ys, yerr=[lo, hi], color=CONDITION_COLOURS[condition], marker="o",
                        capsize=3, linewidth=1.7, label=CONDITION_NAMES[condition])
        ax.set_title(ARCH_TITLES[arch], fontsize=FS + 1)
        ax.set_xticks([0, 1, 2], [SPACE_TITLES[space] for space in SPACES],
                      rotation=20, ha="right", rotation_mode="anchor")
        ax.tick_params(axis="both", labelsize=FS - 1)
        ax.set_xlim(-0.3, 2.4)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylim(min(0.0, rsa_summary["seed_min"].min() - 0.04), 1.06)
    axes[0].set_ylabel("Similarity to input geometry\n(RSA, consonants only)", fontsize=FS)
    _top_legend(fig, axes[0], fontsize=FS)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    _save(fig, out_dir, "rsa_retention_to_input")


# Fig. 11: violator-control discrimination across training.

HW_PANELS = (
    ("natural", "Natural constraints (20 pairs)"),
    ("unnatural", "Unnatural constraints (20 pairs)"),
    ("sonority_seq", "Sonority-sequencing constraints (4 pairs)"),
)


def plot_hw_trajectory_grid(hw_scores: pd.DataFrame, out_dir: Path) -> None:
    """Architectures in rows, pair groups in columns; per-seed means of the group's pairs."""
    FS = 14
    fig, axes = plt.subplots(len(ARCHS), len(HW_PANELS), figsize=(13.5, 10.8), sharex=True, sharey=True)
    for r, arch in enumerate(ARCHS):
        for c, (group, title) in enumerate(HW_PANELS):
            ax = axes[r, c]
            pairs = hw_scores[hw_scores.arch == arch]
            pairs = pairs[pairs["sonority_seq"]] if group == "sonority_seq" else pairs[pairs["naturalness"] == group]
            per_seed = pairs.groupby(["condition", "seed", "epoch"], as_index=False)["delta_mean_token_surprisal"].mean()
            for condition in CONDITIONS:
                cell = per_seed[per_seed["condition"] == condition]
                _require_complete_trajectory(cell, f"hw/{arch}/{group}/{condition}")
                _trajectory_lines(ax, cell, "delta_mean_token_surprisal", condition)
            _zero_line(ax)
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis="both", labelsize=FS - 1)
            if r == 0:
                ax.set_title(title, fontsize=FS + 1)
            if c == 0:
                ax.set_ylabel(f"{ARCH_TITLES[arch]}\n{SURPRISAL_YLABEL}", fontsize=FS)
            if r == len(ARCHS) - 1:
                ax.set_xticks(EPOCH_TICKS)
                ax.set_xlabel("Epoch", fontsize=FS)
    axes[0, 0].set_ylim(-0.35, 2.25)  # covers every per-seed mean of every group
    _top_legend(fig, axes[0, 0], fontsize=FS + 1)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=1.0)
    _save(fig, out_dir, "hw_trajectory_grid")


# Figs. 6 and 7: the behavioural results per model, with the bigram baseline.

def _seed_dot_panel(ax, per_seed: pd.DataFrame, value: str, bigram_value: float | None,
                    *, ylabel: str, title: str, label_bigram: bool) -> None:
    """x: condition within architecture; one jittered dot per seed and a tick at their mean."""
    FS = 12
    rng = np.random.default_rng(0)
    group_width = len(CONDITIONS) + 1
    xpos = {(arch, condition): gi * group_width + ci for gi, arch in enumerate(ARCHS) for ci, condition in enumerate(CONDITIONS)}
    for (arch, condition), x in xpos.items():
        values = per_seed[(per_seed.arch == arch) & (per_seed.condition == condition)][value].to_numpy()
        jitter = rng.uniform(-0.22, 0.22, size=len(values))
        ax.scatter(x + jitter, values, s=11, color=CONDITION_COLOURS[condition], alpha=0.45, linewidth=0)
        ax.hlines(values.mean(), x - 0.36, x + 0.36, color=CONDITION_COLOURS[condition], linewidth=2.4)
    if bigram_value is not None:
        ax.axhline(bigram_value, color="#444444", linewidth=1.2, linestyle=(0, (4, 3)))
        if label_bigram:
            ax.annotate("Bigram (Witten–Bell)", (max(xpos.values()) + 0.5, bigram_value),
                        textcoords="offset points", xytext=(-2, 4), ha="right",
                        fontsize=FS - 2, color="#444444")
    _zero_line(ax)
    ax.set_xticks(list(xpos.values()), [CONDITION_NAMES[condition] for _, condition in xpos],
                  fontsize=FS - 1, rotation=45, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="y", labelsize=FS - 1)
    for gi, arch in enumerate(ARCHS):  # architecture names below the condition labels
        ax.annotate(ARCH_TITLES[arch], xy=(gi * group_width + 1.5, 0), xycoords=ax.get_xaxis_transform(),
                    xytext=(0, -38), textcoords="offset points", ha="center", va="top",
                    fontsize=FS, annotation_clip=False)
    for gi in range(1, len(ARCHS)):
        ax.axvline(gi * group_width - 1, color="#cccccc", linewidth=0.8)
    ax.set_title(title, fontsize=FS + 1)
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(-0.7, max(xpos.values()) + 0.7)


def plot_behaviour_vs_bigram(eval_dir: Path, out_dir: Path) -> None:
    daland = pd.read_csv(eval_dir / "daland_spearman_by_seed.csv")
    bigram = pd.read_csv(eval_dir / "bigram_daland_spearman.csv").set_index("scope").spearman_rho
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.9), sharey=True)
    for ax, (scope, title), label_bigram in zip(
        axes, (("pooled", "All 96 items"), ("Unattested", "Unattested onsets (36 items)")), (False, True)
    ):
        _seed_dot_panel(ax, daland[daland.scope == scope], "spearman_rho", float(bigram[scope]),
                        ylabel="Spearman $\\rho$ with\nhuman 2AFC scores" if scope == "pooled" else "",
                        title=title, label_bigram=label_bigram)
    axes[0].set_ylim(-0.3, 0.9)
    fig.tight_layout()
    _save(fig, out_dir, "daland_vs_bigram")

    hw = pd.read_csv(eval_dir / "hw_summary_by_seed.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.9), sharey=True)
    for ax, naturalness in zip(axes, ("natural", "unnatural")):
        _seed_dot_panel(ax, hw[hw.naturalness == naturalness], "delta_mean", None,
                        ylabel=SURPRISAL_YLABEL if naturalness == "natural" else "",
                        title=f"{naturalness.capitalize()} constraints (20 pairs)", label_bigram=False)
    fig.tight_layout()
    _save(fig, out_dir, "hw_discrimination")


# Figs. 5 and 8: predicted sonority contours on the held-out words.

CONTOUR_CATEGORIES = ["fall4", "fall3", "fall2", "fall1", "level", "rise1", "rise2", "rise3", "rise4", "end"]
# Blues for falls, grey for level, oranges for rises, hatched dark grey for the end of the word.
CONTOUR_COLOURS = dict(zip(CONTOUR_CATEGORIES, ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#969696",
                                                "#fdd0a2", "#fd8d3c", "#d94801", "#7f2704", "#252525"]))
CONTOUR_LABELS = {"fall4": "fall 4", "fall3": "fall 3", "fall2": "fall 2", "fall1": "fall 1", "level": "level",
                  "rise1": "rise 1", "rise2": "rise 2", "rise3": "rise 3", "rise4": "rise 4", "end": "end of word"}
CONTOUR_PANELS = (
    ("onset-1", "Onset,\n1st consonant"),
    ("onset-2+", "Onset,\nlater consonant"),
    ("nucleus", "Nucleus\n(vowel)"),
    ("coda-nonfinal", "Coda,\nnon-final consonant"),
    ("coda-final", "Coda,\nfinal consonant"),
)
CONTOUR_ROWS = (("data", "Data"), ("bigram", "Bigram"), *((c, CONDITION_NAMES[c]) for c in CONDITIONS))


def _stacked_rows(ax, means: pd.DataFrame, *, fs: float, label_min: float) -> None:
    """One 100%-stacked horizontal bar per row of CONTOUR_ROWS; means is indexed by source.
    Segments are inset by a small gap so that neighbouring shades stay distinct; a
    segment narrower than the gap is drawn at full width, so it still shows as a hairline."""
    gap = 0.005
    n = len(CONTOUR_ROWS)
    for r, (source, _) in enumerate(CONTOUR_ROWS):
        y = n - 1 - r
        left = 0.0
        for cat in CONTOUR_CATEGORIES:
            v = float(means.loc[source, cat])
            inset = gap / 2 if v > gap else 0
            ax.barh(y, v - 2 * inset, left=left + inset, height=0.74, color=CONTOUR_COLOURS[cat],
                    linewidth=0, hatch="/////" if cat == "end" else None,
                    edgecolor="#bdbdbd" if cat == "end" else "none")  # the edge colour is the hatch colour
            if v >= label_min:
                dark = cat in ("fall4", "fall3", "rise3", "rise4", "end")
                ax.text(left + v / 2, y, f"{100 * v:.0f}", ha="center", va="center",
                        fontsize=fs - 1, color="white" if dark else "#222222")
            left += v
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xticks([0, 1.0])
    ax.set_xticklabels(["0", "1"], fontsize=fs - 0.5)
    for label, ha in zip(ax.get_xticklabels(), ("left", "right")):
        label.set_horizontalalignment(ha)
    ax.tick_params(axis="y", length=0, pad=1.5)
    ax.tick_params(axis="x", length=2, pad=1.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#999999")
    ax.spines["bottom"].set_linewidth(0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels([label for _, label in CONTOUR_ROWS][::-1], fontsize=fs)


def plot_contour_expectations(eval_dir: Path, out_dir: Path, arch: str) -> None:
    """One panel per syllabic role; rows: the data, the bigram and the four conditions of one architecture."""
    heldout = pd.read_csv(eval_dir / "contour_heldout.csv")
    heldout = heldout[heldout.arch.isin(["data", "bigram", arch])]
    fs = 6.5
    fig, axes = plt.subplots(1, len(CONTOUR_PANELS), figsize=(6.7, 1.75))
    fig.subplots_adjust(left=0.075, right=0.995, top=0.80, bottom=0.30, wspace=0.16)
    for ax, (role, title) in zip(axes, CONTOUR_PANELS):
        means = heldout[heldout.context == role].groupby("source", sort=False)[CONTOUR_CATEGORIES].mean()
        _stacked_rows(ax, means, fs=fs, label_min=0.17)
        ax.set_title(title, fontsize=fs + 0.5, pad=3)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    handles = [Patch(facecolor=CONTOUR_COLOURS[c], hatch="/////" if c == "end" else None,
                     edgecolor="#bdbdbd" if c == "end" else "none", linewidth=0, label=CONTOUR_LABELS[c])
               for c in CONTOUR_CATEGORIES]
    fig.legend(handles=handles, loc="lower center", ncol=10, frameon=False, fontsize=fs,
               bbox_to_anchor=(0.5, -0.02), handlelength=1.3, handleheight=0.9, columnspacing=0.9,
               title="Sonority class of the next phoneme relative to the current one (probability mass)",
               title_fontsize=fs)
    fig.savefig(out_dir / f"contour_expectations_{arch}.png", dpi=300)
    plt.close(fig)
    print("wrote", out_dir / f"contour_expectations_{arch}.png")


# Fig. 10: per-phoneme displacement of the in-context consonants probe.

IPA = {"P": "p", "B": "b", "T": "t", "D": "d", "K": "k", "G": "g", "CH": "tʃ", "JH": "dʒ", "F": "f",
       "V": "v", "TH": "θ", "DH": "ð", "S": "s", "Z": "z", "SH": "ʃ", "ZH": "ʒ", "HH": "h",
       "M": "m", "N": "n", "NG": "ŋ", "L": "l", "R": "r", "W": "w", "Y": "j"}
CLASS_LABELS = {1: "Obstruents", 2: "Nasals", 3: "Liquids", 4: "Glides"}


def plot_heatmap(summary: pd.DataFrame, out_path: Path, consistency_threshold: float = 0.9) -> None:
    """Phonemes (rows, by sonority class) x models (columns): mean displacement out of the
    phoneme's own class block, red positive, blue negative. Dots mark cells in which at
    least consistency_threshold of the seeds are displaced in the same direction."""
    phonemes = summary[["phoneme", "true_rank"]].drop_duplicates().sort_values(["true_rank", "phoneme"])
    order, ranks = phonemes.phoneme.tolist(), phonemes.true_rank.to_numpy()
    columns = [(arch, condition) for arch in ARCHS for condition in CONDITIONS]

    def grid(value: str) -> np.ndarray:
        return summary.pivot(index="phoneme", columns=["arch", "condition"], values=value).loc[order, columns].to_numpy()

    displacement = grid("mean_oob")
    consistent = (grid("p_oob_positive") >= consistency_threshold) | (grid("p_oob_negative") >= consistency_threshold)

    FS = 12  # the paper includes the figure at about half scale
    fig, ax = plt.subplots(figsize=(0.62 * len(columns) + 2.4, 0.27 * len(order) + 1.9))
    vmax = max(np.nanmax(np.abs(displacement)), 1e-9)
    im = ax.imshow(displacement, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_yticks(range(len(order)), [IPA.get(t, t) for t in order], fontsize=FS)
    ax.set_xticks(range(len(columns)), [CONDITION_NAMES[condition] for _, condition in columns], fontsize=FS - 1,
                  rotation=45, ha="right", rotation_mode="anchor")
    n_cond = len(CONDITIONS)
    for gi, arch in enumerate(ARCHS):
        ax.annotate(ARCH_TITLES[arch], xy=(gi * n_cond + (n_cond - 1) / 2, 0),
                    xycoords=ax.get_xaxis_transform(), xytext=(0, -40),
                    textcoords="offset points", ha="center", va="top", fontsize=FS + 1,
                    annotation_clip=False)
    for gi in range(1, len(ARCHS)):
        ax.axvline(gi * n_cond - 0.5, color="black", linewidth=0.8)
    # Lines between the sonority classes, with the class names on the right.
    bounds = [i for i in range(1, len(ranks)) if ranks[i] != ranks[i - 1]]
    for b in bounds:
        ax.axhline(b - 0.5, color="black", linewidth=0.8)
    for start, end in zip([0, *bounds], [*bounds, len(ranks)]):
        ax.text(len(columns) - 0.35, (start + end - 1) / 2, CLASS_LABELS[int(ranks[start])], fontsize=FS - 1,
                rotation=90, ha="left", va="center", color="#333333")
    ax.set_xlim(-0.5, len(columns) + 0.4)
    ys, xs = np.where(consistent)
    ax.scatter(xs, ys, s=9, color="black")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.06)
    cb.ax.tick_params(labelsize=FS - 1)
    cb.set_label("Displacement out of own sonority class (positions)\n"
                 "+ (red): treated as more sonorous than it is\n"
                 "− (blue): treated as less sonorous than it is", fontsize=FS - 1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", type=Path, default=Path("outputs/eval"))
    parser.add_argument("--probe-dir", type=Path, default=Path("outputs/probe"))
    parser.add_argument("--hw-dir", type=Path, default=Path("outputs/hw_trajectory"))
    parser.add_argument("--per-phoneme-dir", type=Path, default=Path("outputs/per_phoneme"))
    parser.add_argument("--out", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--latex-dir", type=Path, default=None, help="also copy the figures here")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    plot_sonority_projection_example(args.out)
    plot_behaviour_vs_bigram(args.eval_dir, args.out)
    for arch in ARCHS:
        plot_contour_expectations(args.eval_dir, args.out, arch)

    summary = pd.read_csv(args.probe_dir / "probe_summary.csv")
    scores = pd.read_csv(args.probe_dir / "probe_scores.csv")
    plot_decomposition(summary, args.out)
    plot_probe_trajectory(scores, "lstm", "cv_binary", "cv_binary_probe_trajectory_notitle", args.out, sem_band=True)
    plot_probe_trajectory(scores, "lstm", "consonants", "consonant_probe_trajectory_notitle", args.out)
    plot_probe_trajectory(scores, "tf2", "consonants", "tf2_consonants_trajectory_notitle", args.out)
    for probe in ("full60", "cv_binary", "consonants"):
        plot_probe_trajectory_grid(scores, probe, f"trajectory_grid_{probe}", args.out)
    plot_rsa_retention(pd.read_csv(args.probe_dir / "rsa_summary.csv"), args.out)
    plot_hw_trajectory_grid(pd.read_csv(args.hw_dir / "hw_trajectory_scores.csv"), args.out)
    plot_heatmap(pd.read_csv(args.per_phoneme_dir / "per_phoneme_summary.csv"), args.out / "per_phoneme_displacement.png")

    if args.latex_dir is not None:
        args.latex_dir.mkdir(parents=True, exist_ok=True)
        for name in PAPER_FIGURES:
            shutil.copy2(args.out / name, args.latex_dir / name)
        print(f"copied {len(PAPER_FIGURES)} figures to {args.latex_dir}")


if __name__ == "__main__":
    main()
