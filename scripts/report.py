#!/usr/bin/env python3
"""report.pdf from results/summary.md and results/figures/*.png (fpdf2).   python scripts/report.py [results] [report.pdf]"""
import os
import sys

from fpdf import FPDF

R = sys.argv[1] if len(sys.argv) > 1 else "results"
OUT = sys.argv[2] if len(sys.argv) > 2 else "report.pdf"

INTRO = """Question. A commodity swap desk quoting calendar-month WTI and Henry Hub swaps by RFQ needs a futures curve to price off, hedge ratios for a book of swaps, a hedge execution policy that respects price impact, and a quoting rule that survives informed flow around the weekly EIA releases. Does the recent research - seasonal kernel-ridge and P-spline curves (Filipovic-Pelger-Ye kernels), nonparametric cross-impact (Hey-Neuman-Tuschmann), impact-aware hedging and the Barzykin-Bergault-Guéant closed-form multi-asset quotes - improve on the desk's classical answers (seasonal Nelson-Siegel-Svensson, front-month stacks and strips, symmetric spreads) when each is tested out of sample on free public data and, for quoting, in an RFQ simulator with an exact P&L decomposition?

Method. Header-only C++ behind one CLI, a FIX 4.4 RFQ gateway with an identical-fills self-test, and a numpy reference implementation. Curve: seasonal NSS, cubic P-spline with a second-difference penalty and a seasonal kernel ridge (Sobolev cubic kernel times a periodic kernel), all linear in prices so leave-one-out repricing, extrapolation and hat-matrix key-rate sensitivities are exact; candidates selected on the first third of the daily builds and scored on the rest; calendar-month average swaps with the NYMEX expiry rules. Risk: PCA of the ladder. Impact: pooled power-law self-impact and ridge cross-impact from an hourly order-flow-imbalance proxy, block-bootstrapped over days, with a cumulative decay curve; hedging policies (front stack, strip, factor-minimum, impact-aware with ablations) compared at equal residual risk on random swap books. Quoting: tiered clients with logistic hit probabilities and a hidden informed direction that drifts the market; symmetric quotes, the closed-form skewed quotes, adverse-selection and calendar widening, an oracle; exact decomposition into spread, inventory, adverse, hedge execution and hedge impact; mark-outs, toxicity routing, anticipatory hedging, risk limits; a replay of the real 2024-2026 days with release-day jumps.

Caveats. There are no free historical full curves: the daily history is the EIA front contracts 1-4 to April 2024 plus two years of listed contracts from Yahoo, so complete curves exist only from July 2026 and every result carries the curve completeness of its day. Client flow is simulated and the impact proxy is order-flow imbalance from hourly bars, not tick data."""

FIGS = [("curve.png", "Curve construction on one day: the three models on the observed contracts, leave-one-out repricing errors by contract, and the key-rate sensitivities of the month-3 calendar swap."),
        ("curvehist.png", "Daily builds: test-period LOO error by candidate with the selection, LOO by curve completeness, rolling daily LOO, and the hedge test (month-3 contract dropped and hedged with each curve's key rates)."),
        ("factors.png", "PCA loadings on the ladder over the recent window, and variance explained by decade on the EIA front contracts."),
        ("impact.png", "Price impact from hourly bars: binned self-impact with the power-law and linear fits, cumulative decay after the window, and the explanatory power of self vs self-plus-cross impact."),
        ("hedge.png", "Hedging random swap books: execution cost per $1M at each policy's residual risk and at the front stack's risk (equal-risk ablations); the EIA-panel hedge test by decade with learned vs curve ratios."),
        ("sim.png", "RFQ simulator: frontier over risk aversion, P&L decomposition and risk-adjusted P&L by quoter and hedger."),
        ("policy.png", "Quote structure: mark-outs by tier and window, and the skew between risk-adding and risk-reducing RFQs."),
        ("events.png", "Replay of the real 2024-2026 days with EIA release jumps: P&L by quoter with CIs and mark-outs around releases."),
        ("events_study.png", "Release-day study on the EIA history: release vs other days, inventory-surprise nowcast and hedging-pressure correlations out of sample with permutation p-values.")]


class PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9); self.set_text_color(120); self.cell(0, 6, "commodity-etrading-stack - curves, impact-aware hedging and RFQ quoting for commodity swaps", align="R"); self.ln(8); self.set_text_color(0)

    def footer(self):
        self.set_y(-12); self.set_font("Helvetica", "", 8); self.set_text_color(120); self.cell(0, 6, f"{self.page_no()}", align="C")


def clean(s):
    return (s.replace("–", "-").replace("—", "-").replace("−", "-").replace("σ", "sigma").replace("β", "beta").replace("η", "eta").replace("Δ", "d").replace("×", "x").replace("≥", ">=").replace("≤", "<=").replace("…", "...")
             .replace("²", "^2").replace("±", "+/-").replace("**", "").replace("`", "").replace("√", "sqrt ").replace("γ", "gamma").replace("δ", "delta").replace("μ", "mu").replace("ρ", "rho").replace("λ", "lambda").replace("κ", "kappa").replace("τ", "tau").replace("→", "->").replace("≈", "~").replace("é", "e"))


def md_table(pdf, rows):
    cols = [c.strip() for c in rows[0].strip("|").split("|")]
    data = [[clean(c.strip()) for c in r.strip("|").split("|")] for r in rows[2:]]
    n = len(cols); w = (pdf.w - 20) / n; widths = [w] * n; fs = 6.5 if n <= 9 else 5.2; cut = 40 if n <= 9 else 22
    pdf.set_font("Helvetica", "B", fs)
    for c, wd in zip(cols, widths):
        pdf.cell(wd, 5, clean(c)[:cut], border=1)
    pdf.ln(5); pdf.set_font("Helvetica", "", fs)
    for r in data:
        if pdf.get_y() > pdf.h - 20:
            pdf.add_page()
        for c, wd in zip(r, widths):
            pdf.cell(wd, 4.5, c[:cut], border=1)
        pdf.ln(4.5)
    pdf.ln(2)


def main():
    pdf = PDF(); pdf.set_auto_page_break(auto=True, margin=15); pdf.add_page()
    pdf.set_font("Helvetica", "B", 16); pdf.cell(0, 10, "Curves, Impact-Aware Hedging and RFQ Quoting for Commodity Swaps", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    for para in INTRO.split("\n\n"):
        pdf.multi_cell(0, 4.5, clean(para)); pdf.ln(2)
    for fn, cap in FIGS:
        p = os.path.join(R, "figures", fn)
        if not os.path.exists(p):
            continue
        if pdf.get_y() > pdf.h - 90:
            pdf.add_page()
        pdf.image(p, w=pdf.w - 20); pdf.set_font("Helvetica", "I", 8); pdf.multi_cell(0, 4, clean(cap)); pdf.ln(3); pdf.set_font("Helvetica", "", 9)
    sm = os.path.join(R, "summary.md")
    if os.path.exists(sm):
        pdf.add_page()
        lines = open(sm, encoding="utf-8").read().splitlines(); i = 0
        while i < len(lines):
            l = lines[i]
            if l.startswith("## "):
                pdf.set_font("Helvetica", "B", 11); pdf.ln(2); pdf.cell(0, 7, clean(l[3:]), new_x="LMARGIN", new_y="NEXT"); pdf.set_font("Helvetica", "", 9); i += 1
            elif l.startswith("### "):
                pdf.set_font("Helvetica", "B", 9); pdf.cell(0, 6, clean(l[4:]), new_x="LMARGIN", new_y="NEXT"); pdf.set_font("Helvetica", "", 9); i += 1
            elif l.startswith("|"):
                j = i
                while j < len(lines) and lines[j].startswith("|"):
                    j += 1
                if j - i >= 2:
                    md_table(pdf, lines[i:j])
                i = j
            elif l.startswith("# "):
                i += 1
            elif l.strip():
                pdf.set_x(pdf.l_margin); pdf.multi_cell(0, 4.5, clean(l.strip())); i += 1
            else:
                i += 1
    pdf.output(OUT); print("wrote", OUT)


if __name__ == "__main__":
    main()
