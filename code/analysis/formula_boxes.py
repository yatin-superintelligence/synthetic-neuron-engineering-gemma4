"""The formulas of the methods section as two typeset images, for the Kaggle page (which shows plain text, not mathematics).

Usage: python tools/analysis/formula_boxes.py
Output: paper/box1_changing_neurons.png, paper/box2_finding_neurons.png (1,640 pixels wide, like the figures).
Each row is a short label and a formula set with matplotlib's STIX mathematics, the typeface of the PDF."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SURF, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e4e3df"
NL = chr(10)
plt.rcParams.update({"font.family": "DejaVu Sans", "mathtext.fontset": "stix"})
W = 8.2  # inches; 1,640 pixels at 200 dpi


def box(path, title, rows, notes, row_h=0.47, tall=()):
    heights = [row_h * (1.45 if i in tall else 1.0) for i in range(len(rows))]
    top, note_h = 0.52, 0.27
    H = top + sum(heights) + note_h * len(notes) + 0.22
    fig = plt.figure(figsize=(W, H), facecolor=SURF)
    fig.text(0.025, 1 - 0.20 / H, title, ha="left", va="center", fontsize=10.5, fontweight="bold", color=INK)
    y = top
    for (label, formula), h in zip(rows, heights):
        fig.add_artist(plt.Line2D([0.025, 0.975], [1 - y / H, 1 - y / H], color=GRID, linewidth=0.8))
        mid = 1 - (y + h / 2) / H
        fig.text(0.025, mid, label, ha="left", va="center", fontsize=9.2, color=INK2, linespacing=1.35)
        fig.text(0.36, mid, formula, ha="left", va="center", fontsize=13, color=INK)
        y += h
    fig.add_artist(plt.Line2D([0.025, 0.975], [1 - y / H, 1 - y / H], color=GRID, linewidth=0.8))
    y += 0.10
    for note in notes:
        fig.text(0.025, 1 - (y + note_h / 2) / H, note, ha="left", va="center", fontsize=8.6, color=INK2)
        y += note_h
    fig.savefig(path, dpi=200)
    plt.close(fig)


box(os.path.join(ROOT, "paper", "box1_changing_neurons.png"), "Box 1. Notation, and the formulas for changing neurons", [
    ("Feed-forward block of a layer", r"$y = W_{\mathrm{down}}\,a, \qquad a = \mathrm{gelu}(W_{\mathrm{gate}}\,x)\odot(W_{\mathrm{up}}\,x)$"),
    ("Into the residual stream; logits", r"$h \leftarrow h + w\odot y\,/\,\mathrm{rms}(y), \qquad \mathrm{logits} = E\,(w_f\odot h\,/\,\mathrm{rms}(h))$"),
    ("Direction removal", r"$W_{\mathrm{down}} \leftarrow (I - v\,v^{\mathrm{T}})\,W_{\mathrm{down}}$"),
    ("Engineered neuron: detector", r"$r = \mathrm{argmin}\ \sum_t\,\omega_t\,(r^{\mathrm{T}}a_t - z_t)^2 + \lambda\,\Vert r\Vert^2$"),
    ("Engineered neuron: write", r"$W_{\mathrm{down}} \leftarrow W_{\mathrm{down}} + u\,r^{\mathrm{T}}$"),
    (r"Condition on the output vector $u$", r"$g^{\mathrm{T}}u = c\cdot\mathrm{rms}(y)\cdot\mathrm{rms}(h), \qquad g = e\odot w\odot w_f, \quad e = E_Y - E_X$"),
    (r"Usual choice of $u$", r"$u \ \propto\ (e\,/\,\Vert e\Vert^2)\oslash(w\odot w_f)$"),
    (r"Shortest $u$ (used on 31B)", r"$u^{*} = c\cdot\mathrm{rms}(y)\cdot\mathrm{rms}(h)\cdot g\,/\,\Vert g\Vert^2$"),
    ("Gated neuron's activation", r"$\mathrm{gelu}\,(10\,(q^{\mathrm{T}}x - 0.5))$"),
    (r"Scaling neuron $i$ by $1 + s$", r"$W_{\mathrm{down}}[:, i] \leftarrow (1+s)\,W_{\mathrm{down}}[:, i]$"),
], [
    r"$x$, $y$: the block's input and output.   $a$: neuron activations.   $h$: residual stream.   $E$: output embeddings.",
    r"$w$, $w_f$: learned norm weights.   $v$: unit direction of looping.   $z_t = 1$ at the target situation, else 0.   $\omega_t$ balances the classes.",
    r"$Y$, $X$: the tokens to raise and to lower by $c$ logits.   $q$: a detector on $x$.   $s = 1$ doubles a neuron, $s = -1$ silences it.",
    r"$\odot$, $\oslash$: element-wise product and division.   rms: root mean square.",
], tall=(3,))

box(os.path.join(ROOT, "paper", "box2_finding_neurons.png"), "Box 2. The formulas for finding causal neurons and for measuring damage", [
    ("Mean log-probability per token" + NL + r"of a file edit $z$ after the context $p$", r"$L(z) = \dfrac{1}{|z|}\ \sum_t\,\log P(z_t \mid p,\ z_{<t})$"),
    ("Preference for the correct fix;" + NL + "the decision gap", r"$\Delta = L(\mathrm{correct}) - L(\mathrm{wrong}), \qquad \mathrm{decision\ gap} = -\Delta$"),
    (r"Causal score of neuron (layer $\ell$, index $i$)",
     r"$\sum_{t\,\in\,\mathrm{correct}} a_{\ell i}\,\dfrac{\partial L(\mathrm{correct})}{\partial a_{\ell i}}\ -\ \sum_{t\,\in\,\mathrm{wrong}} a_{\ell i}\,\dfrac{\partial L(\mathrm{wrong})}{\partial a_{\ell i}}$"),
    ("Damage", r"$D = L(\mathrm{own}) - L'(\mathrm{own}), \qquad \mathrm{reported\ as}\ \ 1 - e^{-D}$"),
], [
    r"correct, wrong: the passing run's file edit and the failing run's own file edit (first 64 tokens of each).",
    r"$a_{\ell i}$: activation of neuron $i$ in layer $\ell$.   own: the last 128 tokens of the failing run's own reasoning.",
    r"$L'$: $L$ with the intervention applied.",
], row_h=0.66, tall=(0, 2))
print("written")
