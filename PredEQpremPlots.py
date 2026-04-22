"""
Equity Premium Composite Predictor Plots (1993–2001 Subsample)
=============================================================

Creates 4 figures:
  1. Positive predictor index – Quarterly
  2. Positive predictor index – Monthly
  3. Negative predictor index – Quarterly
  4. Negative predictor index – Monthly

Each figure shows:
  - Average normalised predictor index (raw + 1-year MA)
  - Log equity premium (raw + 1-year MA, secondary axis)

Usage:
    python plot_predictors.py
    python plot_predictors.py path/to/Data_Seminar.xlsx
"""

import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 0. Config
# ---------------------------------------------------------------------------
FILE = sys.argv[1] if len(sys.argv) > 1 else "Data_Seminar.xlsx"

POS_VARS = ["d/p", "d/y", "e/p", "d/e", "svar", "b/m", "ltr", "tms", "dfy", "dfr"]
NEG_VARS_Q = ["ntis", "tbl", "lty", "infl", "i/k"]
NEG_VARS_M = ["ntis", "tbl", "lty", "infl"]

ALPHA_RAW  = 0.35
LW_RAW     = 0.8
LW_MA      = 1.8

PRED_COLOR = "#2166ac"
EP_COLOR   = "#d6604d"

# ---------------------------------------------------------------------------
# 1. Load & parse dates
# ---------------------------------------------------------------------------
xl = pd.read_excel(FILE, sheet_name=None)
q_raw = xl["Quarterly"].copy()
m_raw = xl["Monthly"].copy()


def parse_quarterly(col):
    year = col // 10
    qtr  = col % 10
    month = (qtr - 1) * 3 + 1
    return pd.to_datetime({"year": year, "month": month, "day": 1})


def parse_monthly(col):
    year  = col // 100
    month = col % 100
    return pd.to_datetime({"year": year, "month": month, "day": 1})


q_raw["date"] = parse_quarterly(q_raw["yyyyq"])
m_raw["date"] = parse_monthly(m_raw["yyyymm"])

q_raw = q_raw.set_index("date").sort_index()
m_raw = m_raw.set_index("date").sort_index()

# ---------------------------------------------------------------------------
# 1.5 Restrict sample: 1993–2001
# ---------------------------------------------------------------------------
start_date = "1994-01-01"
end_date   = "2001-12-31"

q_raw = q_raw.loc[start_date:end_date]
m_raw = m_raw.loc[start_date:end_date]

# ---------------------------------------------------------------------------
# 2. Normalise predictors (z-score)
# ---------------------------------------------------------------------------
def z_score(series):
    return (series - series.mean()) / series.std(ddof=0)


q = q_raw.copy()
m = m_raw.copy()

for col in set(POS_VARS + NEG_VARS_Q):
    if col in q.columns:
        q[col] = z_score(q_raw[col])

for col in set(POS_VARS + NEG_VARS_M):
    if col in m.columns:
        m[col] = z_score(m_raw[col])

# ---------------------------------------------------------------------------
# 3. Construct composite indices
# ---------------------------------------------------------------------------
def make_index(df, variables):
    valid = [v for v in variables if v in df.columns]
    return df[valid].mean(axis=1)


q["pos_index"] = make_index(q, POS_VARS)
q["neg_index"] = make_index(q, NEG_VARS_Q)

m["pos_index"] = make_index(m, POS_VARS)
m["neg_index"] = make_index(m, NEG_VARS_M)

# ---------------------------------------------------------------------------
# 4. Plotting function
# ---------------------------------------------------------------------------
def plot_index(df, index_col, freq_label, window, figname, title):

    ep_col = "log eqprem"

    idx_raw = df[index_col]
    idx_ma  = idx_raw.rolling(window, min_periods=1).mean()

    ep_raw = df[ep_col]
    ep_ma  = ep_raw.rolling(window, min_periods=1).mean()

    fig, ax = plt.subplots(figsize=(10, 4))

    # Predictor index
    ax.plot(df.index, idx_raw, color=PRED_COLOR,
            lw=LW_RAW, alpha=ALPHA_RAW, label="Raw index")
    ax.plot(df.index, idx_ma, color=PRED_COLOR,
            lw=LW_MA, label="1-yr MA index")
    ax.set_ylabel("Average normalised predictor", color=PRED_COLOR)
    ax.tick_params(axis="y", labelcolor=PRED_COLOR)

    # Equity premium
    ax2 = ax.twinx()
    ax2.plot(df.index, ep_raw, color=EP_COLOR,
             lw=LW_RAW, alpha=ALPHA_RAW, label="Raw EqPrem")
    ax2.plot(df.index, ep_ma, color=EP_COLOR,
             lw=LW_MA, label="1-yr MA EqPrem")
    ax2.set_ylabel("Equity Premium", color=EP_COLOR)
    ax2.tick_params(axis="y", labelcolor=EP_COLOR)

    ax.set_title(f"{title} – {freq_label}")
    ax.axhline(0, color="grey", lw=0.5, ls="--")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))  # tighter ticks for short sample
    ax.tick_params(axis="x", rotation=30)

    # Combined legend
    lines = ax.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=8, loc="upper left")

    fig.savefig(figname, dpi=150, bbox_inches="tight")
    print(f"Saved: {figname}")
    plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Produce figures
# ---------------------------------------------------------------------------
plot_index(q, "pos_index", "", 4,
           "positive_index_quarterly2.png",
           "")

plot_index(m, "pos_index", "", 12,
           "positive_index_monthly2.png",
           "")

plot_index(q, "neg_index", "", 4,
           "negative_index_quarterly2.png",
           "")

plot_index(m, "neg_index", "", 12,
           "negative_index_monthly2.png",
           "")

print("\nDone. Four PNG files written to the current directory.")