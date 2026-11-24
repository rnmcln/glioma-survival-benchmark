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
from lifelines import KaplanMeierFitter

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
MLP3 = ["tf_compact", "tf_medium", "tf_regularised"]
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
def figure_s_decision_rule():
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
    ax.axvspan(-config.MEANINGFUL_AUROC_DELTA, config.MEANINGFUL_AUROC_DELTA, color=LIGHT, alpha=0.45, lw=0, zorder=0)
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
    ax.annotate("class-weighted (planned)", (d.loc[first, "delta_lo"], ypos[first] - 0.13),
                xytext=(-4, 2), textcoords="offset points", va="center", ha="right", fontsize=6, color=ACCENT)
    ax.annotate("without class weighting", (du.loc[first, "delta_lo"], ypos[first] + 0.17),
                xytext=(-4, -2), textcoords="offset points", va="center", ha="right", fontsize=6, color=GREY)
    ax.text(config.MEANINGFUL_AUROC_DELTA, 1.0, "+0.03", transform=tx, fontsize=6, va="bottom", ha="center")
    ax.text(-config.MEANINGFUL_AUROC_DELTA, 1.0, "\u22120.03", transform=tx, fontsize=6, va="bottom", ha="center")
    save(fig, "figureS6_decision_rule")


# Supplementary figure S2: discrimination within strata ---------------------------------------------
def figure_s_strata():
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
    save(fig, "figureS2_strata")


def figure3_weighting(stem="figure3_class_weighting"):
    c = pd.read_csv(config.TABLES / "class_weighting.csv")
    c = c[c.setting.str.startswith("int_")]
    fig, (a, b, cc) = plt.subplots(1, 3, figsize=(178 * MM, 60 * MM), gridspec_kw={"wspace": 0.55})
    lim = (-3.0, 1.6)
    a.plot(lim, lim, color=GREY, lw=0.5, ls=(0, (3, 2)))
    a.axhline(0, color=GREY, lw=0.5)
    for m in MLP3:
        for v, filled in (("weighted", True), ("unweighted", False)):
            sub = c[(c.network == m) & (c.version == v)]
            col = ACCENT if v == "weighted" else INK
            mk = {"tf_compact": "^", "tf_medium": "v", "tf_regularised": "D"}[m]
            a.plot(sub.logit_train_prevalence, sub.calibration_intercept, marker=mk, ms=2.6, ls="none",
                   mec=col, mfc=col if filled else "white", mew=0.6, zorder=3)
    a.annotate("expected with weighting", (-1.25, -1.25), xytext=(5, -3), textcoords="offset points",
               fontsize=5.8, color=GREY, rotation=40, ha="left", va="top", rotation_mode="anchor")
    a.annotate("class-weighted", (0.9, 0.95), xytext=(-2, 6), textcoords="offset points", fontsize=6,
               color=ACCENT, ha="right")
    a.annotate("unweighted", (-2.75, 0.1), xytext=(0, 6), textcoords="offset points", fontsize=6, color=INK,
               ha="left")
    a.set_xlim(*lim)
    a.set_ylim(-2.2, 1.6)
    a.set_xlabel("Logit of event rate in training data")
    a.set_ylabel("Calibration intercept")
    panel_label(a, "A")
    rl = pd.read_csv(config.TABLES / "class_weighting_risk_levels.csv")
    b.plot([0, 0.6], [0, 0.6], color=GREY, lw=0.5, ls=(0, (3, 2)))
    hz = {"int_B_1y": ("12 months", "^", 0.0), "int_B_2y": ("24 months", "o", 0.006), "int_B_5y": ("60 months", "s", 0.012)}
    for tag, (lab_, mk, off) in hz.items():
        sub = rl[(rl.setting == tag) & (rl.n_patients >= 5)]
        x = sub.unweighted_level.values + off
        b.vlines(x, sub.weighted_q1, sub.weighted_q3, color=ACCENT, lw=0.6)
        b.plot(x, sub.weighted_median, marker=mk, ms=3, ls="-", lw=0.6, color=ACCENT, mfc="white" if tag == "int_B_5y" else ACCENT)
        last = sub.iloc[-1]
        b.annotate(lab_, (last.unweighted_level + off, last.weighted_median), xytext=(4, 0), textcoords="offset points",
                   fontsize=6, color=ACCENT, va="center")
    b.set_xlim(0, 0.6)
    b.set_ylim(0, 0.8)
    b.set_xticks([0.1, 0.2, 0.4])
    b.set_xlabel("Predicted risk without class weighting")
    b.set_ylabel("Predicted risk with class weighting")
    panel_label(b, "B")
    y, mean, _, _ = load("int_B_1y")
    _, mu, _, _ = load("int_B_1y", config.PREDICTIONS / "unweighted")
    cc.plot([0, 1], [0, 1], color=GREY, lw=0.5, ls=(0, (3, 2)))
    curves = {"penalised_logistic": calibration_curve(y, mean["penalised_logistic"], 10)[:2],
              "tf_compact": calibration_curve(y, mean["tf_compact"], 10)[:2],
              "tf_compact_u": calibration_curve(y, mu["tf_compact"], 10)[:2]}
    sty = {"penalised_logistic": (INK, "o", True, "-"), "tf_compact": (ACCENT, "^", True, "-"),
           "tf_compact_u": (ACCENT, "^", False, (0, (2, 1)))}
    lab = {"penalised_logistic": "Penalised LR", "tf_compact": "Compact MLP, weighted",
           "tf_compact_u": "Compact MLP, unweighted"}
    for m, (mp, ob) in curves.items():
        col, mk, filled, ls = sty[m]
        cc.plot(mp, ob, color=col, lw=0.8, ls=ls)
        cc.plot(mp, ob, marker=mk, ms=2.4, mec=col, mfc=col if filled else "white", mew=0.6, ls="none", zorder=3)
    mp, ob = curves["tf_compact"]
    cc.annotate(lab["tf_compact"], (mp[-3], ob[-3]), xytext=(2, -10), textcoords="offset points", fontsize=6,
                color=ACCENT, ha="left", va="top")
    mp, ob = curves["penalised_logistic"]
    cc.annotate("Penalised LR and\nunweighted MLP", (mp[-2], ob[-2]), xytext=(-6, 4), textcoords="offset points",
                fontsize=6, color=INK, ha="right", va="bottom")
    cc.set_xlim(0, 1)
    cc.set_ylim(0, 1)
    cc.set_xlabel("Predicted probability of death by 1 year")
    cc.set_ylabel("Observed proportion")
    panel_label(cc, "C")
    save(fig, stem)


# Figure 3: external validation ------------------------------------------------------------------
def figure_s_external(stem="figureS5_external_binary"):
    p = pd.read_csv(config.TABLES / "performance_all.csv")
    tags = [("ext_gravendeel_A_2y", "Gravendeel\nset A"), ("ext_gravendeel_B_2y", "Gravendeel\nset B"),
            ("ext_gravendeel_C_2y", "Gravendeel\nset C"), ("ext_rembrandt_B_2y", "REMBRANDT\ngrade,\nhistology"),
            ("ext_rembrandt_C_2y", "REMBRANDT\nset C")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(174 * MM, 66 * MM), gridspec_kw={"wspace": 0.3, "width_ratios": [1.6, 1]})
    w = 0.135
    for i, (t, lab) in enumerate(tags):
        sub = p[p.setting == t].set_index("model")
        for j, m in enumerate(ORDER):
            x = i + (j - (len(ORDER) - 1) / 2) * w
            r = sub.loc[m]
            a.plot([x, x], [r.auroc_lo, r.auroc_hi], color=STYLE[m][0], lw=0.6)
            mark(a, x, r.auroc, m)
    sub = p[p.setting == tags[0][0]].set_index("model")
    for j, m in enumerate(ORDER):
        x = (j - (len(ORDER) - 1) / 2) * w
        a.annotate(LABELS[m], (x, sub[sub.index.isin(ORDER)].auroc_lo.min() - 0.008), rotation=90,
                   ha="center", va="top", fontsize=5.2, color=STYLE[m][0])
    a.set_xticks(range(len(tags)))
    a.set_xticklabels([t[1] for t in tags], fontsize=6)
    a.tick_params(axis="x", length=0)
    a.spines["bottom"].set_visible(False)
    a.set_ylim(0.6, 0.92)
    a.set_xlim(-0.6, len(tags) - 0.4)
    a.set_ylabel("AUROC (95% CI)")
    panel_label(a, "A")
    y, mean, _, _ = load("ext_gravendeel_B_2y")
    _, mu, _, _ = load("ext_gravendeel_B_2y", config.PREDICTIONS / "unweighted")
    b.plot([0, 1], [0, 1], color=GREY, lw=0.5, ls=(0, (3, 2)))
    curves = {"pl": calibration_curve(y, mean["penalised_logistic"], 10)[:2],
              "w": calibration_curve(y, mean["tf_compact"], 10)[:2],
              "u": calibration_curve(y, mu["tf_compact"], 10)[:2]}
    sty = {"pl": (INK, "o", True, "-"), "w": (ACCENT, "^", True, "-"), "u": (ACCENT, "^", False, (0, (2, 1)))}
    for k, (mp, ob) in curves.items():
        col, mk, filled, ls = sty[k]
        b.plot(mp, ob, color=col, lw=0.8, ls=ls)
        b.plot(mp, ob, marker=mk, ms=2.4, mec=col, mfc=col if filled else "white", mew=0.6, ls="none", zorder=3)
    mp, ob = curves["pl"]
    b.annotate("Penalised LR", (mp[4], ob[4]), xytext=(-6, 0), textcoords="offset points", fontsize=6, color=INK,
               ha="right", va="bottom")
    mp, ob = curves["w"]
    b.annotate("Compact MLP,\nweighted", (mp[-2], ob[-2]), xytext=(4, -6), textcoords="offset points", fontsize=6,
               color=ACCENT, ha="left", va="top")
    mp, ob = curves["u"]
    b.annotate("unweighted", (mp[4], ob[4]), xytext=(6, -4), textcoords="offset points", fontsize=6,
               color=ACCENT, ha="left", va="top")
    b.set_xlim(0, 1)
    b.set_ylim(0, 1)
    b.set_xlabel("Predicted probability of death by 2 years")
    b.set_ylabel("Observed proportion")
    panel_label(b, "B")
    save(fig, stem)


def figure_s_dca(stem="figureS3_decision_curves"):
    th = np.array([t for t in config.DCA_THRESHOLDS if 0.02 <= t <= 0.85])
    fig, axes = plt.subplots(1, 2, figsize=(170 * MM, 60 * MM), gridspec_kw={"wspace": 0.6})
    dec = pd.read_csv(config.TABLES / "decision_rule.csv").set_index("setting")
    for ax, (tag, lab) in zip(axes, [("int_B_2y", "TCGA, internal"), ("ext_gravendeel_B_2y", "Gravendeel, external")]):
        y, mean, _, _ = load(tag)
        nn = dec.loc[tag, "best_nn"]
        ax.axvspan(*config.DCA_PLAUSIBLE, color=LIGHT, alpha=0.5, lw=0, zorder=0)
        ax.axhline(0, color=GREY, lw=0.5)
        ta = net_benefit_treat_all(y, th)
        ax.plot(th, ta, color=GREY, lw=0.6, ls=(0, (3, 2)))
        ax.annotate("Treat all", (th[np.argmin(np.abs(ta - 0.03))], 0.03), xytext=(2, 2), textcoords="offset points",
                    fontsize=6, color=GREY)
        ends = {}
        for m in ("penalised_logistic", "hist_gboost", nn):
            nb = net_benefit(y, mean[m], th)
            ax.plot(th, nb, color=STYLE[m][0], lw=0.8, ls="-" if m != "hist_gboost" else (0, (1, 1)))
            ends[m] = nb[-1]
        for k, m in enumerate(sorted(ends, key=ends.get)):
            ax.annotate(LABELS[m], (th[-1], ends[m]), xytext=(3, (k - 1) * 6), textcoords="offset points",
                        fontsize=6, color=STYLE[m][0], va="center")
        ax.set_ylim(-0.05, float(np.mean(y)) + 0.03)
        ax.set_xlim(0, 0.85)
        ax.set_xlabel("Threshold probability")
        ax.set_ylabel("Net benefit")
        ax.text(0.0, 1.02, lab, transform=ax.transAxes, fontsize=6.5, ha="left", va="bottom")
    save(fig, stem)


# Figure 1: primary survival comparison ---------------------------------------------------------------
LBL = {"cox_mlp - cox_flexible": "Network vs flexible Cox", "cox_mlp - cox_linear": "Network vs linear Cox",
       "cox_flexible - cox_linear": "Flexible vs linear Cox"}


def figure1_survival(stem="figure1_survival_comparison"):
    d = pd.read_csv(config.TABLES / "survival_primary_differences.csv")
    e = pd.read_csv(config.TABLES / "survival_external_differences.csv")
    e = e[(e.idh_handling == "mode") & (e.subset == "all")]
    slots = [("TCGA, 12 months", "int", "12"), ("TCGA, 24 months", "int", "24"), ("TCGA, 60 months", "int", "60"),
             ("TCGA, Harrell's C", "int", "c"), ("Gravendeel, 24 months", "gravendeel", "24"),
             ("REMBRANDT, 24 months", "rembrandt", "24")]

    def get(comp, src, h, kind):
        if h == "c":
            if kind == "brier":
                return None
            met = "harrell_c"
        else:
            met = f"{'td_auc' if kind == 'disc' else 'brier'}_{h}"
        if src == "int":
            r = d[(d.comparison == comp) & (d.metric == met)]
        else:
            r = e[(e.target == src) & (e.comparison == comp) & (e.metric == met)]
        return None if r.empty else r.iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 100 * MM), sharey=True, gridspec_kw={"wspace": 0.08})
    ypos, heads, y = [], [], 0
    for comp in LBL:
        heads.append((y, LBL[comp]))
        y += 1
        for lab, src, h in slots:
            ypos.append((comp, lab, src, h, y))
            y += 1
        y += 0.5
    for ax, kind, xl in ((axes[0], "disc", "Difference in time-dependent AUROC or Harrell's C"),
                         (axes[1], "brier", "Difference in IPCW Brier score")):
        for comp, lab, src, h, yy in ypos:
            r = get(comp, src, h, kind)
            if r is None:
                continue
            ext = src != "int"
            col = INK if ext else ACCENT
            ax.plot([r.lo, r.hi], [yy, yy], color=col, lw=0.8)
            ax.plot(r.delta, yy, marker="o", ms=3, color=col, mfc="white" if ext else col, mew=0.7, ls="none", zorder=3)
        ax.axvline(0, color=GREY, lw=0.5)
        ax.set_xlabel(xl)
        ax.set_ylim(y - 0.3, -0.8)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    for yy, h in heads:
        axes[0].text(-0.62, yy, h, transform=axes[0].get_yaxis_transform(), ha="left", va="center", fontsize=6.5,
                     style="italic")
    axes[0].set_yticks([p[-1] for p in ypos])
    axes[0].set_yticklabels([p[1] for p in ypos])
    axes[1].text(0.02, 0.005, "negative: lower error for the first model", transform=axes[1].transAxes, fontsize=5.8,
                 color=INK)
    panel_label(axes[0], "A")
    panel_label(axes[1], "B")
    save(fig, stem)


# Figure 2: discrimination across and within molecular groups ------------------------------------------
def figure2_heterogeneity(stem="figure2_diagnostic_heterogeneity"):
    pr = pd.read_csv(config.TABLES / "survival_group_pairs.csv").set_index("model")
    wg = pd.read_csv(config.TABLES / "survival_within_group.csv")
    gr = pd.read_csv(config.TABLES / "survival_groups.csv")
    models = [("group_only", "Molecular group only"), ("age_and_group", "Age and molecular group"),
              ("cox_linear", "Linear Cox"), ("cox_flexible", "Flexible Cox"), ("cox_mlp", "Network")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(174 * MM, 64 * MM), gridspec_kw={"wspace": 0.55, "width_ratios": [1.1, 1]})
    cols = [("c_between", "Between groups"), ("c_overall", "All pairs"), ("c_within", "Within groups")]
    for k, (m, lab) in enumerate(models[::-1]):
        for j, (c, _) in enumerate(cols):
            v = pr.loc[m, c]
            col = ACCENT if m in ("cox_mlp",) else (GREY if m in ("group_only", "age_and_group") else INK)
            a.plot(v, k, marker=["s", "o", "D"][j], ms=3.2, color=col, mfc="white" if j == 1 else col, ls="none")
        a.plot([pr.loc[m, "c_within"], pr.loc[m, "c_between"]], [k, k], color=LIGHT, lw=1.5, zorder=0)
    a.set_yticks(range(len(models)))
    a.set_yticklabels([lab for _, lab in models[::-1]])
    top = len(models) - 1
    for j, (c, lab) in enumerate(cols):
        a.annotate(lab, (pr.loc[models[0][0], c], top), xytext=((3, 7) if c == "c_overall" else (0, 7)),
                   textcoords="offset points", ha=("right" if c == "c_overall" else "center"), fontsize=6, color=INK)
    a.axvline(0.5, color=GREY, lw=0.5, ls=(0, (3, 2)))
    a.set_xlim(0.4, 1.0)
    a.set_ylim(-0.6, top + 0.9)
    a.set_xlabel("Harrell's C, TCGA cross-validation")
    a.tick_params(axis="y", length=0)
    a.spines["left"].set_visible(False)
    panel_label(a, "A")
    groups = [("IDH-wildtype", "idhwt", 24.0, "IDH-wildtype\n24 months"),
              ("IDH-mutant, non-codeleted", "idhmut_noncodel", 60.0, "IDH-mutant,\nnon-codeleted\n60 months")]
    procs = [("cox_linear", "Linear Cox", INK, "o"), ("cox_flexible", "Flexible Cox", INK, "s"), ("cox_mlp", "Network", ACCENT, "^")]
    for i, (g, key, h, lab) in enumerate(groups):
        whole = gr[(gr.group == g) & (gr.horizon == h)].iloc[0]
        within = wg[(wg.group == key) & (wg.row == "model")].set_index("model")
        for j, (m, ml, col, mk) in enumerate(procs):
            x = i + (j - 1) * 0.12
            b.plot(x - 0.03, whole[f"td_auc_{m}"], marker=mk, ms=3, color=col, mfc="white", mew=0.7, ls="none")
            b.plot(x + 0.03, within.loc[m, "td_auc"], marker=mk, ms=3, color=col, ls="none")
    b.set_xticks(range(len(groups)))
    b.set_xticklabels([g[3] for g in groups], fontsize=6.5)
    b.tick_params(axis="x", length=0)
    b.spines["bottom"].set_visible(False)
    b.axhline(0.5, color=GREY, lw=0.5, ls=(0, (3, 2)))
    b.set_ylim(0.45, 0.8)
    b.set_xlim(-0.5, len(groups) - 0.5)
    b.set_ylabel("Time-dependent AUROC within group")
    b.text(0.02, 0.97, "open: developed in all patients\nfilled: developed within the group\n"
           "circle, linear Cox; square, flexible Cox;\ntriangle, network",
           transform=b.transAxes, fontsize=5.8, va="top", color=INK)
    b.set_ylim(0.40, 0.85)
    panel_label(b, "B")
    save(fig, stem)


def figure_s_survival_calibration(stem="figureS7_survival_calibration"):
    from src.harmonise import load_cohort
    tcga = load_cohort("tcga", with_expr=False).reset_index(drop=True)
    sd = config.PREDICTIONS / "survival"
    fig, axes = plt.subplots(1, 2, figsize=(150 * MM, 66 * MM), gridspec_kw={"wspace": 0.35})
    sty = {"cox_linear": (INK, "o", "Linear Cox"), "cox_flexible": (GREY, "s", "Flexible Cox"), "cox_mlp": (ACCENT, "^", "Network")}
    z0 = np.load(sd / "internal_r0.npz")
    zg = np.load(sd / "external_gravendeel.npz")
    g = load_cohort("gravendeel", with_expr=False).reset_index(drop=True)
    for ax, (d, getr, lab) in zip(axes, [(tcga, lambda m: z0[f"risk__{m}"][:, 1], "TCGA, first repeat"),
                                         (g, lambda m: zg[f"risk__{m}__mode"][:, 1], "Gravendeel")]):
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.5, ls=(0, (3, 2)))
        for m, (col, mk, nm) in sty.items():
            r = getr(m)
            order = np.argsort(r, kind="mergesort")
            px, ob = [], []
            for bidx in np.array_split(order, 8):
                km = KaplanMeierFitter().fit(d.os_months.values[bidx], d.os_event.values[bidx])
                px.append(r[bidx].mean())
                ob.append(1 - float(km.survival_function_at_times(24.0).iloc[0]))
            ax.plot(px, ob, color=col, lw=0.8, marker=mk, ms=2.6)
            k = list(sty).index(m)
            ax.plot(0.60, 0.20 - 0.07 * k, marker=mk, ms=2.6, color=col, transform=ax.transAxes, clip_on=False)
            ax.text(0.63, 0.20 - 0.07 * k, nm, transform=ax.transAxes, fontsize=6, color=col, va="center")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted risk of death by 24 months")
        ax.set_ylabel("Kaplan-Meier risk by 24 months")
        ax.text(0.0, 1.02, lab, transform=ax.transAxes, fontsize=6.5)
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
    for f in config.FIGURES.glob("figure*"):
        f.unlink()
    figure1_survival()
    figure2_heterogeneity()
    figure3_weighting()
    figure_s_calibration()
    figure_s_strata()
    figure_s_dca()
    figure_s_learning()
    figure_s_external()
    figure_s_decision_rule()
    figure_s_survival_calibration()
