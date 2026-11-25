"""Plot the registered contextual RSA portfolio for the research report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAMES = {
    "text_bge": "Source text / BGE", "text_qwen": "Source text / Qwen3",
    "text_gemini": "Source text / Gemini", "image_titan": "Frame / Titan",
    "image_voyage": "Frame / Voyage", "image_gemini": "Frame / Gemini",
    "audio_caption_bge": "Audio caption / BGE", "audio_caption_qwen": "Audio caption / Qwen3",
    "image_caption_bge": "Frame caption / BGE", "video_caption_bge": "Video caption / BGE",
    "video_caption_qwen": "Video caption / Qwen3", "audio_signal": "Audio signal",
    "video_motion": "Motion signal", "video_static": "Frame-average signal",
}
FIGURE_COLORS = {"text-mediated": "#a14b24", "fixed signal": "#6b657a", "direct input": "#205b6d"}


def representation_type(pair: dict) -> str:
    if "caption" in pair["left"] or "caption" in pair["right"]:
        return "text-mediated"
    if pair["interpretation"] == "fixed_signal_control":
        return "fixed signal"
    return "direct input"


def plot(report: Path, output: Path) -> None:
    pairs = [p for p in json.loads(report.read_text(encoding="utf8"))["pairs"] if p["status"] == "measured"]
    if not pairs:
        raise ValueError("no comparisons to plot")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": "#26323b",
                         "text.color": "#26323b", "savefig.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 2.3 + .27 * len(pairs)), sharey=True,
                             gridspec_kw={"width_ratios": [1.12, 1]})
    left, right = axes
    lower = -.25
    common_counts = {len(row["concepts"]) for row in pairs}
    show_row_counts = len(common_counts) > 1
    for i, row in enumerate(pairs):
        position = len(pairs) - i - 1
        category = representation_type(row)
        color = FIGURE_COLORS[category]
        a = row["outcomes"]["paired"]["global_rsa"]["estimate"]
        b = row["outcomes"]["cross_source"]["global_rsa"]["estimate"]
        low, high = row["intervals"]["cross_source"]
        lower = min(lower, low - .05, a - .05)
        left.plot([low, high], [position + .11, position + .11], color=color, alpha=.65, linewidth=1.5)
        left.scatter(b, position + .11, s=27, color=color, zorder=3)
        left.scatter(a, position - .11, s=21, facecolor="white", edgecolor=color, linewidth=1, marker="D", zorder=4)
        within = row["outcomes"]["cross_source"]["within_rsa"]
        if within is not None:
            lower = min(lower, within["estimate"] - .05, within["null_median"] - .05)
            right.plot([within["null_median"], within["null_q95"]], [position, position],
                       color="#aab1b7", linewidth=2, alpha=.75)
            right.scatter(within["estimate"], position, s=27, color=color, zorder=3)
        if show_row_counts:
            right.text(1.04, position, f"{len(row['concepts'])}", transform=right.get_yaxis_transform(),
                       va="center", color="#44525b", fontsize=8)
    left.set_yticks(range(len(pairs)), [
        f"{NAMES[row['left']]}  ×  {NAMES[row['right']]}" for row in reversed(pairs)
    ])
    right.tick_params(axis="y", length=0)
    for axis in axes:
        axis.axvline(0, color="#80909b", linewidth=.7, zorder=0)
        axis.grid(axis="x", color="#e6eaed", linewidth=.6)
        axis.set_xlim(max(-1.02, lower), 1.02)
        axis.set_xlabel("Spearman RSA")
        axis.set_ylim(-.7, len(pairs) - .3)
    left.set_title("All concept relationships")
    right.set_title("Within-group relationships")
    if show_row_counts:
        right.text(1.04, len(pairs) - .08, "Concepts", transform=right.get_yaxis_transform(),
                   va="bottom", color="#44525b", fontsize=8)
    label = "" if show_row_counts else f" · {next(iter(common_counts))} concepts"
    fig.suptitle("Wikimedia Commons contextual geometry" + label,
                 fontsize=15, fontweight="bold", x=.5, y=.98)
    fig.text(.28, .045, "Filled circles  Creator-disjoint folds   •   Open diamonds  Same recordings   •   "
             "Left bars  Creator-weight sensitivity   •   Right bars  Null median to 95th percentile", fontsize=8)
    for i, (label, color) in enumerate(FIGURE_COLORS.items()):
        fig.text(.29 + .132 * i, .017, f"●  {label}", color=color, fontsize=8)
    fig.subplots_adjust(left=.34, right=.95, top=.87, bottom=.12, wspace=.19)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("figure", type=Path)
    args = parser.parse_args()
    plot(args.report, args.figure)
