"""Paper figures, from numbers recorded in DECISIONS.md and tools/analysis outputs (no competition content).

Every figure carries its own title, plain axis labels and direct value labels, so it can be read without the text.
Figure 1: how the agent's attempts end, E2B against E4B (tools/analysis/localization.py; 1,968 and 986 attempts).
Figure 2: file edits that applied at 461 recorded moments just before E2B's first file edit: Test 1, with the start of its reasoning
          forced (EXP-116, D-133); Test 2, with a weight change fitted to stand in for the sentence, its control fitted to the
          neutral sentence, and the sentence itself (EXP-120 and EXP-121, D-141; measured separately, own baseline).
Figure 3: 31B on the 15 tasks it solves only sometimes: tasks solved per run for the baseline and for the two kinds of neuron
          intervention (tools/analysis/exp161_compare.py; D-221, D-223, D-229), and the share of the decision gap each causal
          intervention closes against its damage (EXP-162b dose.json; D-226).
Output: paper/fig1_run_endings.png, paper/fig2_thought_openings.png, paper/fig3_31b_neurons.png.
Palette: the dataviz skill's validated categorical slots (light surface)."""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SURF, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3, S4, PALE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#a8c8ee"
NL = chr(10)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURF, "axes.facecolor": SURF})


def clean(ax, grid="x"):
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    (ax.xaxis if grid == "x" else ax.yaxis).grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# Figure 1: how the agent's attempts end
cats = ["Solved (hidden tests pass)", "Changed the right lines, but wrongly", "Changed other code", "Submitted no patch"]
cols = [S1, S3, S4, S2]
data = {"E2B, the smallest model" + NL + "(1,968 attempts)": [6.6, 24.7, 33.4, 35.3], "E4B" + NL + "(986 attempts)": [17.4, 33.6, 42.4, 6.6]}
fig, ax = plt.subplots(figsize=(8.2, 3.2))
for row, (name, vals) in enumerate(data.items()):
    left = 0.0
    for v, col in zip(vals, cols):
        ax.barh(row, v - 0.4, left=left + 0.2, height=0.56, color=col, linewidth=0)
        if v >= 5:
            ax.text(left + v / 2, row, f"{v:.0f}%", ha="center", va="center", color=INK, fontsize=9)
        left += v
ax.set_yticks(range(len(data)), list(data))
ax.invert_yaxis()
ax.set_xlim(0, 100)
ax.set_xlabel("Share of attempts (%). One attempt is one bug-fix task in one run.")
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in cols], cats, ncol=2, frameon=False, fontsize=8.5, loc="upper center",
          bbox_to_anchor=(0.42, 1.36), handlelength=1.0, labelcolor=INK)
ax.set_title("How the agent's attempts end", loc="left", fontsize=10.5, fontweight="bold", color=INK, pad=44)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "paper", "fig1_run_endings.png"), dpi=200)

# Figure 2: file edits that applied, of 461 moments just before E2B's first file edit (EXP-116, D-133; EXP-120 and EXP-121, D-141)
HI = "The quoting sentence"
groups2 = [("Test 1.  The agent's reasoning is made to begin with a sentence",
            [("No sentence (baseline)", 153, PALE), ("A neutral sentence", 159, PALE), (HI, 221, S1)]),
           ("Test 2.  No sentence is given. Instead, one layer's weights are changed" + NL
            + "so that the model's internal state is as if it had written the sentence",
            [("No weight change (baseline)", 141, PALE),
             ("Weight change for the quoting sentence, layer 10", 147, PALE),
             ("Weight change for the quoting sentence, layer 14", 135, PALE),
             ("Control: weight change for the neutral sentence", 163, PALE),
             ("For comparison: the quoting sentence itself", 220, S1)])]
fig, ax = plt.subplots(figsize=(8.6, 5.3))
fig.subplots_adjust(left=0.47, right=0.975, top=0.845, bottom=0.235)
HX = -(0.47 - 0.02) / (0.975 - 0.47)  # group headings start at the figure's left edge
ys, labels, y = [], [], 0.0
for head, arms in groups2:
    ax.text(HX, y + (0.25 if NL in head else 0), head, transform=ax.get_yaxis_transform(), ha="left", va="center", color=INK,
            fontsize=9.2, fontweight="bold", linespacing=1.35, bbox=dict(facecolor=SURF, edgecolor="none", pad=1.5))
    y += 1.0 + (0.5 if NL in head else 0)
    for name, v, col in arms:
        ax.barh(y, v, height=0.62, color=col, linewidth=0)
        ax.text(v + 3, y, f"{v} of 461", va="center", color=INK, fontsize=9)
        ys.append(y)
        labels.append(name)
        y += 1
    y += 0.55
ax.set_yticks(ys, labels)
ax.set_ylim(y - 1.0, -0.65)
ax.set_xlim(0, 290)
ax.set_xticks(range(0, 251, 50))
ax.set_xlabel("Number of the 461 moments at which the agent's file edit applied")
clean(ax)
fig.text(0.02, 0.975, "A sentence at the start of the agent's reasoning makes more of its file edits apply." + NL
         + "A weight change built to stand in for that sentence does not.", ha="left", va="top", fontsize=10.5, fontweight="bold",
         color=INK, linespacing=1.4)
fig.text(0.02, 0.022, "A file edit applies when the code it quotes matches the file exactly." + NL
         + "The quoting sentence: the agent commits to quote, character for character, the lines it is about to change." + NL
         + "The 461 moments are recorded points just before the smallest model's first file edit; the two tests were run separately.",
         ha="left", va="bottom", fontsize=8.3, color=INK2, linespacing=1.4)
fig.savefig(os.path.join(ROOT, "paper", "fig2_thought_openings.png"), dpi=200)

# Figure 3: 31B on the 15 tasks it solves only sometimes
runs = [("Baseline model" + NL + "(6 runs)", [11, 11, 10, 11, 13, 9], INK2),
        ("Predictive neurons" + NL + "changed (6 runs)", [9, 11, 9, 10, 9, 9], S4),
        ("Causal neurons" + NL + "changed (2 runs)", [9, 10], S1)]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.5), gridspec_kw={"width_ratios": [1.05, 1]})
a1.axvspan(9, 13, color="#ecebe6", zorder=0)
a1.text(11, -0.72, "baseline range", ha="center", va="center", color=INK2, fontsize=8.5)
for i, (name, vals, col) in enumerate(runs):
    seen = {}
    for v in sorted(vals):
        k = seen.get(v, 0)
        a1.scatter(v, i + 0.15 * k - 0.15 * (vals.count(v) - 1) / 2, s=52, color=col, edgecolor=SURF, linewidth=1.5, zorder=3)
        seen[v] = k + 1
    m = sum(vals) / len(vals)
    a1.plot([m, m], [i - 0.34, i + 0.34], color=col, linewidth=2, zorder=2)
a1.set_yticks(range(len(runs)), [r[0] for r in runs])
a1.set_xlim(7.5, 14.5)
a1.set_xticks(range(8, 15))
a1.set_ylim(2.6, -1.0)
a1.set_xlabel("Tasks solved in a run, out of 15" + NL + "(each dot is one run; the bar is the average)")
clean(a1)
a1.set_title("Tasks solved did not rise", loc="left", fontsize=10, fontweight="bold", color=INK)
GAP = 0.4565  # nats per token by which the wrong fix leads, mean of 25 held-out cases (EXP-162b)
dose = [("64 neurons", 0.2007, 0.0502, "below"), ("64 neurons," + NL + "4x stronger", 0.2482, 0.3383, "below"),
        ("256 neurons", 0.2599, 0.0956, "above"), ("256 neurons," + NL + "4x stronger", 0.3245, 0.7943, "aboveleft")]
for name, c, d, where in dose:
    x, yv = 100 * (1 - math.exp(-d)), 100 * c / GAP
    a2.scatter(x, yv, s=52, color=S1, edgecolor=SURF, linewidth=1.5, zorder=3)
    if where == "above":
        a2.text(x + 1.2, yv + 3, name, va="bottom", color=INK, fontsize=8.5)
    elif where == "aboveleft":
        a2.text(x + 1.5, yv + 4, name, ha="right", va="bottom", color=INK, fontsize=8.5)
    else:
        a2.text(x + 1.5, yv - 2, name, va="top", color=INK, fontsize=8.5)
a2.scatter([0], [0.5], s=52, color=PALE, edgecolor=SURF, linewidth=1.5, zorder=3)
a2.text(2, 3, "random neurons", va="bottom", color=INK2, fontsize=8.5)
a2.axhline(100, color=INK2, linewidth=1, linestyle=(0, (3, 3)))
a2.text(59, 97, "100%: the correct fix becomes the preferred one", ha="right", va="top", color=INK2, fontsize=8.5)
a2.set_xlim(-2, 60)
a2.set_ylim(-5, 110)
a2.set_xlabel("Damage: how much less likely the model finds" + NL + "its own ordinary text (% per token)")
a2.set_ylabel("Decision gap closed (%)")
clean(a2, grid="y")
a2.spines["left"].set_visible(False)
a2.set_title("Gap closed, against damage done", loc="left", fontsize=10, fontweight="bold", color=INK)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "paper", "fig3_31b_neurons.png"), dpi=200)
print("written")
