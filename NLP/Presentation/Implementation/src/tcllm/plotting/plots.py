"""Figures for the report, built with matplotlib.

Colour rules followed here (validated against a CVD checker, not eyeballed):

* Light surface ``#fcfcfb`` throughout; figures are static PNGs for the report.
* Categorical hues are used in fixed slot order and never cycled:
  slot 1 blue ``#2a78d6``, slot 2 orange ``#eb6834``, slot 3 aqua ``#1baf7a``.
  Every chart here uses at most three series, the set that passes the
  all-pairs colour-vision-deficiency check.
* Sequential encoding (the confusion matrices) is ONE hue, light -> dark blue.
  Never a rainbow.
* Text wears text tokens (primary/secondary/muted ink), never a series colour.
  A coloured mark beside the text carries identity.
* Grid and spines are recessive; marks are thin; every bar and line end is
  directly labelled so identity is never carried by colour alone.

The paper's own English numbers are drawn as *reference* guides only, in muted
ink with a dashed stroke, and are never encoded as a series colour. See
``docs/deviations.md`` for why they are not directly comparable to ours.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: no display on this machine, and we always write files

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from ..eval.metrics import PAPER_ENGLISH_SENTIMENT, Result

# --------------------------------------------------------------------------- #
# tokens
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BAND = "#f0efec"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # slots 1-3, all-pairs validated

# One hue, light -> dark, for magnitude.
BLUE_RAMP = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)
SEQ_BLUE = LinearSegmentedColormap.from_list("tcllm_blue", BLUE_RAMP)

DPI = 200

# Minimum vertical separation, in points, between two displaced end labels on a
# line chart. Two lines of text at 8.5pt need roughly this much room.
LABEL_GAP_POINTS = 30


def _style(fig, axes: Iterable) -> None:
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="x", color=GRID, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
            ax.spines[side].set_linewidth(0.8)
        ax.tick_params(colors=INK_MUTED, labelsize=8.5, length=3, width=0.8)


def _save(fig, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 1. our methods, one bar each
# --------------------------------------------------------------------------- #
def plot_method_comparison(
    results: Sequence[Result],
    out_path: str | Path,
    *,
    title: str = "English SENTIMENT: our methods on the official test split",
) -> Path:
    """Accuracy per method, sorted, with the paper's English range behind it.

    One measure, one series, so the bars share a single hue and no legend box is
    needed - the y labels name the methods. Macro-F1 is printed beside each bar
    rather than added as a second mark, which keeps the chart to one axis.
    """
    ordered = sorted(results, key=lambda r: r.accuracy)
    names = [r.method for r in ordered]
    values = [r.accuracy for r in ordered]

    fig, ax = plt.subplots(figsize=(9.6, 0.62 * len(ordered) + 2.2))
    _style(fig, [ax])

    paper = sorted(PAPER_ENGLISH_SENTIMENT.values())
    low, high = paper[0], paper[-1]
    ax.axvspan(low, high, color=BAND, zorder=0)
    ax.axvline(
        PAPER_ENGLISH_SENTIMENT["GPT-4 (zero-shot)"],
        color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1,
    )

    positions = range(len(ordered))
    ax.barh(
        list(positions), values, height=0.68, color=SERIES[0],
        edgecolor=SURFACE, linewidth=0.0, zorder=3,
    )

    for y, result in zip(positions, ordered):
        ax.text(
            result.accuracy + 0.008, y, f"{result.accuracy:.3f}",
            va="center", ha="left", fontsize=8.5, color=INK, zorder=4,
        )
        ax.text(
            result.accuracy + 0.062, y, f"F1 {result.macro_f1:.3f}",
            va="center", ha="left", fontsize=8, color=INK_SECONDARY, zorder=4,
        )

    ax.set_yticks(list(positions))
    ax.set_yticklabels(names, fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Accuracy on the official test split (870 tweets)", fontsize=9, color=INK_SECONDARY)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=16)

    axis_top = len(ordered) - 0.35
    ax.text(
        PAPER_ENGLISH_SENTIMENT["GPT-4 (zero-shot)"], axis_top,
        "  best paper result, 0.71\n  (GPT-4 zero-shot)",
        fontsize=8, color=INK_MUTED, va="top", ha="left",
    )
    # Footnote, not an in-axes label: at this bar count anything placed under the
    # bars lands on the x tick labels.
    fig.text(
        0.01, 0.005,
        "Shaded band = the range of the paper's own English results (0.51-0.71). "
        "Dashed line = their best, 0.71.  Our bars are cheap stand-ins, not reruns of their models.",
        fontsize=8, color=INK_MUTED, ha="left", va="bottom",
    )
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# 2. our number against the paper's, paired
# --------------------------------------------------------------------------- #
def plot_paper_vs_ours(
    pairs: Sequence[dict],
    out_path: str | Path,
    *,
    title: str = "Our stand-ins vs the paper's English results",
) -> Path:
    """Grouped bars: one pair per method family.

    Two series, so a legend is mandatory - and every bar is direct-labelled as
    well, so the pairing survives greyscale printing.
    """
    labels = [p["family"] for p in pairs]
    ours = [p["ours"] for p in pairs]
    paper = [p["paper"] for p in pairs]

    fig, ax = plt.subplots(figsize=(9.6, 0.95 * len(pairs) + 2.6))
    _style(fig, [ax])
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")

    height = 0.34
    ys = list(range(len(pairs)))
    ax.barh([y + height / 2 + 0.02 for y in ys], ours, height=height,
            color=SERIES[0], edgecolor=SURFACE, linewidth=0.0, zorder=3,
            label="This work (measured here)")
    ax.barh([y - height / 2 - 0.02 for y in ys], paper, height=height,
            color=SERIES[1], edgecolor=SURFACE, linewidth=0.0, zorder=3,
            label="Paper (authors' reported numbers)")

    for y, value in zip(ys, ours):
        ax.text(value + 0.008, y + height / 2 + 0.02, f"{value:.3f}",
                va="center", ha="left", fontsize=8.5, color=INK, zorder=4)
    for y, value in zip(ys, paper):
        ax.text(value + 0.008, y - height / 2 - 0.02, f"{value:.3f}",
                va="center", ha="left", fontsize=8.5, color=INK, zorder=4)

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9, color=INK_SECONDARY)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("Accuracy (English SENTIMENT, official test split)", fontsize=9, color=INK_SECONDARY)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=16)
    ax.legend(
        loc="upper right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY,
        handlelength=1.1, handleheight=0.9,
    )
    # Footnote at figure level: inside the axes it collides with the bottom bars.
    fig.text(
        0.01, 0.005,
        "Different methods: our bars are cheap stand-ins (TF-IDF / seed lexicon / templates), "
        "not reruns of the paper's models.",
        fontsize=8, color=INK_MUTED, ha="left", va="bottom",
    )
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# 3. learning curve
# --------------------------------------------------------------------------- #
def plot_learning_curve(
    rows: Sequence[dict],
    out_path: str | Path,
    *,
    metric: str = "accuracy",
    title: str = "How many labelled examples does each classifier need?",
) -> Path:
    """Metric against training examples per class, one line per classifier.

    Three series -> direct labels at the right end plus a legend, so identity is
    never carried by colour alone. The paper's few-shot and supervised English
    results are drawn as dashed reference guides in muted ink.
    """
    methods = []
    for row in rows:
        if row["method"] not in methods:
            methods.append(row["method"])
    if len(methods) > len(SERIES):
        raise ValueError(f"{len(methods)} series would exceed the validated categorical slots ({len(SERIES)})")

    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    _style(fig, [ax])
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)

    series: list[tuple[str, list, list]] = []

    for index, method in enumerate(methods):
        points = sorted((r for r in rows if r["method"] == method), key=lambda r: r["n_per_label"])
        xs = [p["n_per_label"] for p in points]
        ys = [p[metric] for p in points]
        ax.plot(xs, ys, color=SERIES[index], linewidth=2.0, marker="o", markersize=5.5,
                markeredgecolor=SURFACE, markeredgewidth=1.4, zorder=3, label=method)
        series.append((method, xs, ys))

    # These lines converge, so naive end labels land on top of each other. Push
    # them apart vertically and draw a hairline leader back to the data point.
    order = sorted(range(len(series)), key=lambda i: series[i][2][-1])
    placed: dict[int, float] = {}
    previous = None
    for i in order:
        y = series[i][2][-1]
        if previous is not None and y - previous < LABEL_GAP_POINTS:
            y = previous + LABEL_GAP_POINTS
        placed[i] = y
        previous = y

    for i, (method, xs, ys) in enumerate(series):
        offset = placed[i] - ys[-1]
        ax.annotate(
            f"{method}\n{ys[-1]:.3f}",
            xy=(xs[-1], ys[-1]),
            xytext=(10, offset),
            textcoords="offset points",
            fontsize=8.5, color=INK_SECONDARY, va="center", ha="left",
            arrowprops=None if abs(offset) < 1e-9 else dict(
                arrowstyle="-", color=BASELINE, linewidth=0.7, shrinkA=0, shrinkB=2,
            ),
        )

    guides = [
        ("FastFit few-shot, paper 0.59", PAPER_ENGLISH_SENTIMENT["FastFit (few-shot, 10/label)"]),
        ("GTE + LogReg on real data, paper 0.65",
         PAPER_ENGLISH_SENTIMENT["Logistic Reg. on GTE (real)"]),
    ]
    for label, value in guides:
        ax.axhline(value, color=INK_MUTED, linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
        # Anchored left: the curves sit near 0.35 there, so the reference labels
        # have clear space, whereas the right side carries the end labels.
        ax.text(0.005, value + 0.006, label, transform=ax.get_yaxis_transform(),
                fontsize=8, color=INK_MUTED, ha="left", va="bottom")

    ax.set_xscale("log")
    ax.set_xticks([10, 50, 100, 200, 500, 1839])
    ax.set_xticklabels(["10", "50", "100", "200", "500", "1839\n(all)"])
    ax.set_xlim(8, 2400)
    ax.set_ylim(0.2, 0.85)
    ax.set_xlabel("Labelled training examples per class (drawn from the train split only)",
                  fontsize=9, color=INK_SECONDARY)
    ax.set_ylabel(metric.replace("_", "-").capitalize(), fontsize=9, color=INK_SECONDARY)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=14)
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# 4. confusion matrices
# --------------------------------------------------------------------------- #
def plot_confusion_matrices(
    items: Sequence[dict],
    labels: Sequence[str],
    out_path: str | Path,
    *,
    title: str = "Where each method goes wrong (official test split, 870 tweets)",
) -> Path:
    """Small multiples of row-normalised confusion matrices, one blue ramp.

    Rows are normalised so the three panels share a single 0-1 colour scale; raw
    counts are printed in each cell so the numbers are readable without the colour.
    """
    labels = list(labels)
    count = len(items)
    fig, axes = plt.subplots(1, count, figsize=(4.1 * count, 4.3))
    # Without extra spacing a panel's y tick labels sit against the previous
    # panel's right edge.
    fig.subplots_adjust(wspace=0.45)
    if count == 1:
        axes = [axes]

    for ax, item in zip(axes, items):
        matrix = item["confusion"]
        totals = [sum(row) for row in matrix]
        normalised = [
            [cell / totals[i] if totals[i] else 0.0 for cell in row]
            for i, row in enumerate(matrix)
        ]
        ax.imshow(normalised, cmap=SEQ_BLUE, vmin=0.0, vmax=1.0, aspect="auto")
        ax.set_facecolor(SURFACE)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8.5, color=INK_SECONDARY, rotation=30, ha="right")
        ax.set_yticklabels(labels, fontsize=8.5, color=INK_SECONDARY)
        ax.tick_params(colors=INK_MUTED, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        for i, row in enumerate(normalised):
            for j, value in enumerate(row):
                ax.text(
                    j, i, f"{matrix[i][j]}\n{value * 100:.0f}%",
                    ha="center", va="center", fontsize=8,
                    color="#ffffff" if value > 0.55 else INK,
                )
        ax.set_xlabel("predicted", fontsize=8.5, color=INK_MUTED)
        ax.set_title(
            f"{item['title']}\naccuracy {item['accuracy']:.3f}",
            fontsize=9.5, color=INK, pad=8,
        )

    axes[0].set_ylabel("true label", fontsize=8.5, color=INK_MUTED)
    fig.suptitle(title, fontsize=11.5, color=INK, x=0.02, ha="left", y=1.02)
    return _save(fig, out_path)


# --------------------------------------------------------------------------- #
# 5. per-class F1
# --------------------------------------------------------------------------- #
def plot_per_class_f1(
    results: Sequence[Result],
    labels: Sequence[str],
    out_path: str | Path,
    *,
    title: str = "Per-class F1: which sentiment is hardest?",
) -> Path:
    """Grouped bars, one series per method, one group per class.

    Ends directly labelled, legend present, single y-axis.
    """
    labels = list(labels)
    if len(results) > len(SERIES):
        raise ValueError(f"{len(results)} series would exceed the validated categorical slots")

    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    _style(fig, [ax])
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)

    group_width = 0.78
    bar_width = group_width / len(results)
    xs = list(range(len(labels)))

    for index, result in enumerate(results):
        offset = (index - (len(results) - 1) / 2) * bar_width
        heights = [result.per_class[label]["f1"] for label in labels]
        positions = [x + offset for x in xs]
        ax.bar(positions, heights, width=bar_width * 0.88, color=SERIES[index],
               edgecolor=SURFACE, linewidth=0.0, zorder=3, label=result.method)
        for x, height in zip(positions, heights):
            ax.text(x, height + 0.012, f"{height:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK_SECONDARY, zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9, color=INK_SECONDARY)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1", fontsize=9, color=INK_SECONDARY)
    ax.set_title(title, fontsize=11.5, color=INK, loc="left", pad=14)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY,
              ncols=min(len(results), 3))
    return _save(fig, out_path)


__all__ = [
    "BLUE_RAMP",
    "SERIES",
    "plot_confusion_matrices",
    "plot_learning_curve",
    "plot_method_comparison",
    "plot_paper_vs_ours",
    "plot_per_class_f1",
]
