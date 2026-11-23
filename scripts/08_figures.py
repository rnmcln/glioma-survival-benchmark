"""Figures (vector PDF/SVG and 600-dpi TIFF) written to results/figures/.

Visual encoding: transparent regression models in near-black, neural networks
in a single muted blue, other models in light grey; model class is also coded
by marker shape and identified by direct labels.
"""
import _path  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from src import config
from src.cv import load
from src.evaluation import calibration_curve, net_benefit, net_benefit_treat_all
from src.models import LABELS
from src.utils import load_json

INK, ACCENT, GREY, LIGHT = "#2E3440", "#5E81AC", "#A3ABB9", "#D8DEE9"
plt.rcParams.update({
    "font.family": "Inter", "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "axes.linewidth": 0.5, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.labelcolor": INK, "text.color": INK,
    "lines.linewidth": 0.8, "pdf.fonttype": 42, "svg.fonttype": "none", "legend.frameon": False,
})
MM = 1 / 25.4
STYLE = {  # colour, marker, filled
    "logistic_age_grade": (INK, "o", False), "penalised_logistic": (INK, "o", True),
    "random_forest": (GREY, "s", False), "hist_gboost": (GREY, "s", True),
    "tf_compact": (ACCENT, "^", False), "tf_medium": (ACCENT, "v", False),
    "tf_regularised": (ACCENT, "^", True), "ft_transformer": (ACCENT, "D", False),
}
ORDER = ["logistic_age_grade", "penalised_logistic", "random_forest", "hist_gboost",
         "tf_compact", "tf_medium", "tf_regularised"]


def save(fig, stem):
    for ext in ("pdf", "svg"):
        fig.savefig(config.FIGURES / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(config.FIGURES / f"{stem}.tif", dpi=600, bbox_inches="tight", pad_inches=0.02,
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(config.FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def mark(ax, x, y, m, size=3.2, **kw):
    c, mk, filled = STYLE[m]
    ax.plot(x, y, marker=mk, ms=size, mec=c, mfc=c if filled else "white", mew=0.7,
            ls="none", zorder=3, **kw)


def direct_labels(ax, curves):
    """Label each curve at the interior point where it is furthest from the
    other curves, above or below according to its position."""
    for m, (x, yv) in curves.items():
        best, where = -1, None
        for i in range(1, len(x) - 1):
            others = [np.interp(x[i], *curves[o]) for o in curves if o != m]
            gap = min(abs(yv[i] - v) for v in others)
            if gap > best:
                best, where = gap, (i, yv[i] >= max(others))
        i, above = where
        ax.annotate(LABELS[m], (x[i], yv[i]), xytext=(0, 6 if above else -7), textcoords="offset points",
                    ha="center", va="bottom" if above else "top", fontsize=6, color=STYLE[m][0])


def panel_label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, va="bottom", ha="right")


# Figure 1: decision-rule forest ------------------------------------------------
def figure1():
    d = pd.read_csv(config.TABLES / "decision_rule.csv")
    du = pd.read_csv(config.TABLES / "decision_rule_unweighted.csv").set_index("setting")
    groups = [("Internal validation, TCGA", [s for s in d.setting if s.startswith("int_") and not
                                             any(k in s for k in ("idh", "gbm"))]),
              ("Within molecular or histological strata, TCGA", [s for s in d.setting if any(k in s for k in ("idh", "gbm"))]),
              ("External validation", [s for s in d.setting if s.startswith("ext_")])]
    d = d.set_index("setting")
    ypos, labels, y, heads = {}, [], 0, []
    for g, tags in groups:
        heads.append((y, g))
        y += 1
        for t in tags:
            ypos[t] = y
            y += 1
        y += 0.4
    fig, ax = plt.subplots(figsize=(160 * MM, 0.255 * y + 0.5))
    ax.axvline(0, color=GREY, lw=0.5, zorder=1)
    ax.axvline(config.MEANINGFUL_AUROC_DELTA, color=INK, lw=0.5, ls=(0, (3, 2)), zorder=1)
    for t, yy in ypos.items():
        r = d.loc[t]
        ax.plot([r.delta_lo, r.delta_hi], [yy - 0.13] * 2, color=ACCENT, lw=0.8)
        ax.plot(r.delta_auroc, yy - 0.13, marker="o", ms=3.2, color=ACCENT, ls="none", zorder=3)
        if t in du.index:
            u = du.loc[t]
            ax.plot([u.delta_lo, u.delta_hi], [yy + 0.17] * 2, color=GREY, lw=0.8)
            ax.plot(u.delta_auroc, yy + 0.17, marker="o", ms=3.2, mfc="white", mec=GREY, mew=0.7,
                    ls="none", zorder=3)
    xmin = min(d.delta_lo.min(), du.delta_lo.min()) - 0.01
    xmax = max(d.delta_hi.max(), du.delta_hi.max()) + 0.01
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(y - 0.2, -0.8)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([d.loc[t, "label"].replace("TCGA, ", "").replace("Gravendeel, ", "Gravendeel: ")
                        .replace("REMBRANDT, ", "REMBRANDT: ") for t in ypos])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for yy, g in heads:
        ax.text(xmin, yy, g, ha="left", va="center", fontsize=7, color=INK, style="italic")
    ax.set_xlabel("Difference in AUROC, best neural network minus best transparent model (95% CI)")
    # right-hand text columns
    tx = ax.get_xaxis_transform()
    ax.text(1.03, -0.8, "Calibration\nslope", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5)
    ax.text(1.15, -0.8, "Criterion", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5)
    for t, yy in ypos.items():
        r = d.loc[t]
        ax.text(1.03, yy, f"{r.slope_nn:.2f}", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5)
        ax.text(1.15, yy, "met" if r.criterion_met else "not met", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=6.5)
    first = list(ypos)[0]
    ax.annotate("class-weighted (prespecified)", (d.loc[first, "delta_lo"], ypos[first] - 0.13),
                xytext=(-4, 2), textcoords="offset points", va="center", ha="right", fontsize=6, color=ACCENT)
    ax.annotate("without class weighting", (du.loc[first, "delta_lo"], ypos[first] + 0.17),
                xytext=(-4, -2), textcoords="offset points", va="center", ha="right", fontsize=6, color=GREY)
    ax.text(config.MEANINGFUL_AUROC_DELTA, 1.0, "margin 0.03", transform=tx, fontsize=6, va="bottom", ha="center")
    save(fig, "figure1_decision_rule")


# Figure 2: discrimination within strata ---------------------------------------------
def figure2():
    p = pd.read_csv(config.TABLES / "performance_all.csv")
    strata = [("int_B_2y", "All diffuse gliomas\n2-year"), ("int_idhwt_2y", "IDH-wildtype\n2-year"),
              ("int_idhwt_1y", "IDH-wildtype\n1-year"), ("int_idhmut_5y", "IDH-mutant\n5-year"),
              ("int_gbm_1y", "Glioblastoma\n1-year")]
    fig, ax = plt.subplots(figsize=(150 * MM, 62 * MM))
    w = 0.11
    for i, (tag, lab) in enumerate(strata):
        sub = p[p.setting == tag].set_index("model")
        for j, m in enumerate(ORDER):
            x = i + (j - (len(ORDER) - 1) / 2) * w
            r = sub.loc[m]
            ax.plot([x, x], [r.auroc_lo, r.auroc_hi], color=STYLE[m][0], lw=0.6)
            mark(ax, x, r.auroc, m)
        n, e = int(sub.iloc[0].n), int(sub.iloc[0].events)
        ax.text(i, 0.40, f"n = {n}\n{e} deaths", ha="center", va="bottom", fontsize=6, color=INK)
    ax.axhline(0.5, color=GREY, lw=0.5, ls=(0, (3, 2)))
    ax.set_xticks(range(len(strata)))
    ax.set_xticklabels([s[1] for s in strata])
    ax.tick_params(axis="x", length=0)
    ax.spines["bottom"].set_visible(False)
    ax.set_ylim(0.38, 1.0)
    ax.set_xlim(-0.6, len(strata) - 0.4)
    ax.set_ylabel("AUROC (95% CI)")
    # direct labels on the first group
    sub = p[p.setting == "int_B_2y"].set_index("model")
    for j, m in enumerate(ORDER):
        x = (j - (len(ORDER) - 1) / 2) * w
        ax.annotate(LABELS[m], (x, sub.loc[m].auroc_lo), xytext=(0, -3), textcoords="offset points",
                    rotation=90, ha="center", va="top", fontsize=5.5, color=STYLE[m][0])
    save(fig, "figure2_strata")


# Figure 3: calibration and net benefit in Gravendeel --------------------------------------
def figure3(tag="ext_gravendeel_B_2y", stem="figure3_calibration_dca"):
    y, mean, _, _ = load(tag)
    dec = pd.read_csv(config.TABLES / "decision_rule.csv").set_index("setting")
    nn = dec.loc[tag, "best_nn"]
    models = ["penalised_logistic", "hist_gboost", nn]
    fig, (a, b) = plt.subplots(1, 2, figsize=(150 * MM, 66 * MM), gridspec_kw={"wspace": 0.42})
    a.plot([0, 1], [0, 1], color=GREY, lw=0.5, ls=(0, (3, 2)))
    curves = {m: calibration_curve(y, mean[m], 10)[:2] for m in models}
    for m in models:
        mp, ob = curves[m]
        a.plot(mp, ob, color=STYLE[m][0], lw=0.8)
        mark(a, mp, ob, m, size=2.6)
    direct_labels(a, curves)
    # distribution of predicted risk (penalised LR) as a rug
    for v in mean["penalised_logistic"]:
        a.plot([v, v], [-0.06, -0.03], color=INK, lw=0.25, alpha=0.4)
    a.set_xlim(0, 1.0)
    a.set_ylim(-0.07, 1.0)
    a.set_xlabel("Predicted probability of death by 2 years")
    a.set_ylabel("Observed proportion")
    panel_label(a, "A")
    th = np.array([t for t in config.DCA_THRESHOLDS if 0.02 <= t <= 0.85])
    b.axvspan(*config.DCA_PLAUSIBLE, color=LIGHT, alpha=0.5, lw=0, zorder=0)
    b.axhline(0, color=GREY, lw=0.5)
    ta = net_benefit_treat_all(y, th)
    b.plot(th, ta, color=GREY, lw=0.6, ls=(0, (3, 2)))
    b.annotate("Treat all", (th[np.argmin(np.abs(ta - 0.05))], 0.05), xytext=(2, 2), textcoords="offset points",
               fontsize=6, color=GREY)
    ends = {}
    for m in models:
        nb = net_benefit(y, mean[m], th)
        b.plot(th, nb, color=STYLE[m][0], lw=0.8, ls="-" if m != "hist_gboost" else (0, (1, 1)))
        ends[m] = nb[-1]
    for k, m in enumerate(sorted(ends, key=ends.get)):
        b.annotate(LABELS[m], (th[-1], ends[m]), xytext=(3, (k - 1) * 6), textcoords="offset points",
                   fontsize=6, color=STYLE[m][0], va="center")
    prev = float(np.mean(y))
    b.set_ylim(-0.05, prev + 0.03)
    b.set_xlim(0, 0.85)
    b.set_xlabel("Threshold probability")
    b.set_ylabel("Net benefit")
    panel_label(b, "B")
    save(fig, stem)


# Supplementary figures -------------------------------------------------------------------
def figure_s_calibration(tag="int_B_2y", stem="figureS1_calibration_internal"):
    y, mean, _, _ = load(tag)
    ms = ORDER
    fig, axes = plt.subplots(2, 4, figsize=(170 * MM, 85 * MM), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.35, "wspace": 0.15})
    for ax, m in zip(axes.ravel(), ms):
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.5, ls=(0, (3, 2)))
        mp, ob, _ = calibration_curve(y, mean[m], 10)
        ax.plot(mp, ob, color=STYLE[m][0], lw=0.8)
        mark(ax, mp, ob, m, size=2.4)
        ax.text(0.03, 0.97, LABELS[m], transform=ax.transAxes, va="top", fontsize=6.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticks([0, 0.5, 1])
        ax.set_yticklabels(["0", "0.5", "1"])
    axes.ravel()[-1].axis("off")
    fig.supxlabel("Predicted probability of death by 2 years", fontsize=7)
    fig.supylabel("Observed proportion", fontsize=7)
    save(fig, stem)


def figure_s_cohorts(stem="figureS2_auroc_by_cohort"):
    p = pd.read_csv(config.TABLES / "performance_all.csv")
    tags = [("int_B_2y", "TCGA, internal"), ("ext_gravendeel_B_2y", "Gravendeel, external"),
            ("ext_rembrandt_B_2y", "REMBRANDT, external")]
    fig, axes = plt.subplots(1, 3, figsize=(170 * MM, 55 * MM), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (t, lab) in zip(axes, tags):
        sub = p[p.setting == t].set_index("model")
        for k, m in enumerate(ORDER[::-1]):
            r = sub.loc[m]
            ax.plot([r.auroc_lo, r.auroc_hi], [k, k], color=STYLE[m][0], lw=0.7)
            mark(ax, r.auroc, k, m)
        ax.text(0.0, 1.02, lab, transform=ax.transAxes, fontsize=6.5, ha="left", va="bottom")
        ax.set_xlabel("AUROC (95% CI)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(ORDER)))
    axes[0].set_yticklabels([LABELS[m] for m in ORDER[::-1]])
    save(fig, stem)


def figure_s_dca_internal(tag="int_B_2y", stem="figureS3_decision_internal"):
    y, mean, _, _ = load(tag)
    dec = pd.read_csv(config.TABLES / "decision_rule.csv").set_index("setting")
    nn = dec.loc[tag, "best_nn"]
    th = np.array([t for t in config.DCA_THRESHOLDS if 0.02 <= t <= 0.85])
    fig, b = plt.subplots(figsize=(85 * MM, 62 * MM))
    b.axvspan(*config.DCA_PLAUSIBLE, color=LIGHT, alpha=0.5, lw=0, zorder=0)
    b.axhline(0, color=GREY, lw=0.5)
    b.plot(th, net_benefit_treat_all(y, th), color=GREY, lw=0.6, ls=(0, (3, 2)))
    ends = {}
    for m in ("penalised_logistic", "hist_gboost", nn):
        nb = net_benefit(y, mean[m], th)
        b.plot(th, nb, color=STYLE[m][0], lw=0.8, ls="-" if m != "hist_gboost" else (0, (1, 1)))
        ends[m] = nb[-1]
    for k, m in enumerate(sorted(ends, key=ends.get)):
        b.annotate(LABELS[m], (th[-1], ends[m]), xytext=(3, (k - 1) * 6), textcoords="offset points",
                   fontsize=6, color=STYLE[m][0], va="center")
    b.set_ylim(-0.05, float(np.mean(y)) + 0.03)
    b.set_xlim(0, 0.85)
    b.set_xlabel("Threshold probability")
    b.set_ylabel("Net benefit")
    save(fig, stem)


def figure_s_learning(stem="figureS4_learning_curves"):
    h = load_json(config.METRICS / "final_network_histories.json")
    fig, axes = plt.subplots(1, 3, figsize=(170 * MM, 50 * MM), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, arch in zip(axes, ("compact", "medium", "regularised")):
        tr, va = np.array(h[arch]["loss"]), np.array(h[arch]["val_loss"])
        ep = np.arange(1, len(tr) + 1)
        ax.plot(ep, tr, color=INK, lw=0.7)
        ax.plot(ep, va, color=ACCENT, lw=0.7)
        best = int(np.argmin(va)) + 1
        ax.axvline(best, color=GREY, lw=0.5, ls=(0, (3, 2)))
        k = max(1, len(ep) // 6)
        ax.annotate("training", (ep[k], tr[k]), xytext=(4, 10), textcoords="offset points", fontsize=6, color=INK)
        ax.annotate("validation", (ep[k], va[k]), xytext=(0, -12), textcoords="offset points", fontsize=6,
                    color=ACCENT, va="top")
        ax.text(0.97, 0.97, f"{LABELS['tf_' + arch]}\nbest epoch {best}", transform=ax.transAxes,
                ha="right", va="top", fontsize=6.5)
        ax.set_xlabel("Epoch")
    axes[0].set_ylabel("Binary cross-entropy")
    save(fig, stem)


if __name__ == "__main__":
    figure1()
    figure2()
    figure3()
    figure_s_calibration()
    figure_s_cohorts()
    figure_s_dca_internal()
    figure_s_learning()
