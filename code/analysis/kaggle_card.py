"""The card image for the Kaggle Writeup form (the form asks for 560 x 280 pixels).

Usage: python tools/analysis/kaggle_card.py
Output: paper/kaggle_version/card_560x280.png and card_1120x560.png (the same picture at twice the size).
The three bars are the shares of the decision gap closed in the dose test (EXP-162b dose.json; D-226): 64 causal neurons 44%,
256 neurons 57%, 256 neurons four times stronger 71%. Colours as in the paper's figures."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "paper", "kaggle_version")
SURF, INK, INK2, GRID, S1, PALE = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#a8c8ee"
NL = chr(10)
plt.rcParams.update({"font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(5.6, 2.8), facecolor=SURF)
fig.text(0.05, 0.86, "Synthetic Neuron Engineering" + NL + "of a Gemma 4 Coding Agent", ha="left", va="top", fontsize=12.6,
         fontweight="bold", color=INK, linespacing=1.3)
fig.text(0.05, 0.575, "Predictive Neurons, Causal Neurons," + NL + "and the Decision Gap", ha="left", va="top", fontsize=9.4,
         fontweight="bold", color=INK, linespacing=1.35)
fig.text(0.05, 0.33, "Training-free weight changes steer" + NL + "a local coding agent.", ha="left", va="top", fontsize=8.4,
         color=INK2, linespacing=1.45)
ax = fig.add_axes([0.635, 0.20, 0.32, 0.50], facecolor=SURF)
vals = [44, 57, 71]
names = ["64", "256", "256," + NL + "4x stronger"]
ax.bar(range(3), vals, width=0.62, color=[PALE, PALE, S1], linewidth=0)
for i, v in enumerate(vals):
    ax.text(i, v + 3, f"{v}%", ha="center", va="bottom", fontsize=8, color=INK)
ax.axhline(100, color=INK2, linewidth=0.9, linestyle=(0, (3, 3)))
ax.text(2.42, 103, "100%: correct fix preferred", ha="right", va="bottom", fontsize=6.6, color=INK2)
ax.set_ylim(0, 118)
ax.set_xlim(-0.55, 2.55)
ax.set_xticks(range(3), names, fontsize=6.8, color=INK2)
ax.set_yticks([])
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(axis="x", length=0)
fig.text(0.635, 0.86, "Share of the decision gap closed" + NL + "by scaling causal neurons", ha="left", va="top", fontsize=7.6,
         fontweight="bold", color=INK, linespacing=1.3)
os.makedirs(OUT, exist_ok=True)
fig.savefig(os.path.join(OUT, "card_560x280.png"), dpi=100)
fig.savefig(os.path.join(OUT, "card_1120x560.png"), dpi=200)
print("written")
