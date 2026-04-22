"""
OLS Rolling-Window Out-of-Sample Forecasts: eqprem ~ a + b*x_i + e_i

For each predictor x_i, a rolling OLS window is estimated on the most recent
W observations ending at t-1, and the fitted value is forecast for period t.

Also computes:
  - 1/N combination forecast (equal-weighted average across all predictor forecasts)
  - OOS R² = 1 - SS_res(model) / SS_res(hist_avg) over three sub-periods:
      Full  : 1979-2022
      Early : 1979-1993
      Late  : 1994-2022
  - Clark-West (CW) test statistic and one-sided p-value for each forecast
    vs the historical average benchmark.
    H0: model does not improve on benchmark (one-sided, reject for large CW stat)

Clark-West statistic (Clark & West, 2007):
  f_t = (e0_t^2 - e1_t^2) + (yhat1_t - yhat0_t)^2
  CW  = mean(f_t) / se(f_t)  [t-stat from regressing f_t on a constant]
  p   = P(N(0,1) > CW)  [one-sided]

Window sizes : 5, 10, 15 years
First OOS period : 1979Q1 (quarterly) / 1979M1 (monthly)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Configuration ────────────────────────────────────────────────────────────

FILE_PATH   = "Data_Seminar.xlsx"
OUTPUT_PATH = "OLS_Forecasts_RollingWindow.xlsx"
TARGET      = "log eqprem"

WINDOW_YEARS = [5, 10, 15]

OOS_START_Q = 19791    # 1979 Q1
OOS_START_M = 197901   # 1979 Jan

# Earlier start for S_fr rolling windows — needs 15y history before 1979
OOS_START_Q_EXT = 19651   # 1965 Q1
OOS_START_M_EXT = 196501  # 1965 Jan

# Sub-period boundaries (inclusive on both ends, applied to date codes)
PERIODS_Q = {
    "1979-2022": (19791,  20224),
    "1979-1993": (19791,  19934),
    "1994-2022": (19941,  20224),
}
PERIODS_M = {
    "1979-2022": (197901, 202212),
    "1979-1993": (197901, 199312),
    "1994-2022": (199401, 202212),
}

PREDICTORS_Q = [
    "d/p", "d/y", "e/p", "d/e", "svar", "b/m", "ntis",
    "tbl", "lty", "ltr", "tms", "dfy", "dfr", "infl", "i/k"
]

PREDICTORS_M = [
    "d/p", "d/y", "e/p", "d/e", "svar", "b/m", "ntis",
    "tbl", "lty", "ltr", "tms", "dfy", "dfr", "infl"
    # i/k excluded: entirely missing in Monthly sheet
]


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_data(file_path):
    sheets = pd.read_excel(file_path, sheet_name=None)
    df_q = sheets["Quarterly"].copy()
    df_m = sheets["Monthly"].copy()
    df_q.rename(columns={"simple eqprem": "simple_eqprem"}, inplace=True)
    df_m.rename(columns={"simple eqprem": "simple_eqprem"}, inplace=True)
    return df_q, df_m


# ── Historical Average Benchmark ─────────────────────────────────────────────

def expanding_historical_avg(df, date_col, oos_start):
    """
    For each OOS period t, compute the expanding-window mean of eqprem
    using only observations strictly before t. This is the benchmark for OOS R².
    """
    df = df.reset_index(drop=True)
    oos_idx = df.index[df[date_col] >= oos_start].tolist()

    hist_avg = pd.Series(np.nan, index=df.index)
    for t in oos_idx:
        past = df.loc[:t - 1, TARGET].dropna()
        if len(past) > 0:
            hist_avg.at[t] = past.mean()

    return hist_avg


# ── Rolling-Window OOS Forecast ───────────────────────────────────────────────

def rolling_oos_forecast(df, predictor, date_col, oos_start, window):
    """
    For each period t >= oos_start:
      1. Lag the predictor by 1 period (pred_{t-1} predicts eqprem_t).
      2. Fit OLS on the rolling window [t-window, t-1] of complete cases.
      3. Predict eqprem at period t using pred_{t-1}.
    """
    data = df[[date_col, TARGET, predictor]].copy().reset_index(drop=True)
    lag_col = f"{predictor}_lag1"
    data[lag_col] = data[predictor].shift(1)
    data = data.dropna(subset=[lag_col]).reset_index(drop=True)

    oos_idx = data.index[data[date_col] >= oos_start].tolist()
    forecasts = pd.Series(np.nan, index=data.index)

    for t in oos_idx:
        train = data.iloc[max(0, t - window): t][[lag_col, TARGET]].dropna()

        if len(train) < 3:
            continue

        X_train = sm.add_constant(train[lag_col], has_constant="add")
        y_train = train[TARGET]

        try:
            model = sm.OLS(y_train, X_train).fit()
        except Exception:
            continue

        x_t = data.loc[t, lag_col]
        if pd.isna(x_t):
            continue

        forecasts.at[t] = model.params["const"] + model.params[lag_col] * x_t

    # Re-index back onto the original df index via date alignment
    date_to_forecast = dict(zip(data[date_col], forecasts))
    result = df[date_col].map(date_to_forecast)
    result.index = df.index
    return result


# ── OOS R² ────────────────────────────────────────────────────────────────────

def oos_r2(actual, forecast, benchmark):
    """
    OOS R² = 1 - sum((actual - forecast)^2) / sum((actual - benchmark)^2)
    Computed over periods where all three series are non-missing.
    """
    valid = actual.notna() & forecast.notna() & benchmark.notna()
    ss_model = ((actual[valid] - forecast[valid]) ** 2).sum()
    ss_bench = ((actual[valid] - benchmark[valid]) ** 2).sum()
    if ss_bench == 0:
        return np.nan
    return 1 - ss_model / ss_bench


# ── Clark-West Test ───────────────────────────────────────────────────────────

def clark_west(actual, forecast, benchmark):
    """
    Clark & West (2007) one-sided test (matching ols_expanding_window.py).

    f_t = (e0_t^2 - e1_t^2) + (yhat1_t - yhat0_t)^2
        where e0_t = actual - benchmark,  e1_t = actual - forecast

    CW statistic = f.mean() / (f.std(ddof=1) / sqrt(n))  [simple t-test]
    p-value is one-sided upper tail of Student-t(n-1).

    Returns (cw_stat, p_value). NaN if insufficient data.
    """
    valid = actual.notna() & forecast.notna() & benchmark.notna()
    if valid.sum() < 5:
        return np.nan, np.nan

    y  = actual[valid].values
    y0 = benchmark[valid].values
    y1 = forecast[valid].values

    e0 = y - y0
    e1 = y - y1
    f  = (e0 ** 2) - (e1 ** 2) + (y1 - y0) ** 2

    n       = len(f)
    cw_stat = f.mean() / (f.std(ddof=1) / np.sqrt(n))
    p_value = stats.t.sf(cw_stat, df=n - 1)   # one-sided upper tail

    return float(cw_stat), float(p_value)


def cw_subperiod(out_df, date_col, forecast_col, period_bounds):
    """Run Clark-West test restricted to a date sub-period."""
    lo, hi = period_bounds
    mask = (out_df[date_col] >= lo) & (out_df[date_col] <= hi)
    sub  = out_df[mask]
    return clark_west(sub["eqprem_actual"], sub[forecast_col], sub["hist_avg_benchmark"])


def oos_r2_subperiod(out_df, date_col, forecast_col, period_bounds):
    lo, hi = period_bounds
    mask = (out_df[date_col] >= lo) & (out_df[date_col] <= hi)
    sub  = out_df[mask]
    return oos_r2(sub["eqprem_actual"], sub[forecast_col], sub["hist_avg_benchmark"])


def expanding_historical_avg_simple(df, date_col, oos_start, target="simple_eqprem"):
    df = df.reset_index(drop=True)
    oos_idx = df.index[df[date_col] >= oos_start].tolist()
    hist_avg = pd.Series(np.nan, index=df.index)
    for t in oos_idx:
        past = df.loc[:t - 1, target].dropna()
        if len(past) > 0:
            hist_avg.at[t] = past.mean()
    return hist_avg


def lagged_rolling_variance(df, target_col, window):
    df = df.reset_index(drop=True)
    return df[target_col].rolling(window, min_periods=window // 2).var(ddof=1).shift(1)


def _weights(forecast_col: str, df: pd.DataFrame, gamma: float,
             w_min: float = 0.0, w_max: float = 1.5) -> pd.Series:
    return (df[forecast_col] / (gamma * df["sigma2"])) .clip(w_min, w_max)


def _cer(w: pd.Series, r: pd.Series, ann: int, gamma: float) -> float:
    port = w * r
    return float(ann * port.mean() - (gamma / 2) * ann * port.var(ddof=1))


def _cer_gain_bps(forecast_col: str, df: pd.DataFrame, ann: int, gamma: float) -> float:
    valid = df[["simple_eqprem", "sigma2", forecast_col, "predicted_HM"]].dropna()
    if len(valid) < 24:
        return np.nan
    w_model = _weights(forecast_col, valid, gamma)
    w_bench = _weights("predicted_HM", valid, gamma)
    return (_cer(w_model, valid["simple_eqprem"], ann, gamma)
            - _cer(w_bench, valid["simple_eqprem"], ann, gamma)) * 10_000


def _stationary_bootstrap_indices(n: int, B: int, rng, avg_block_len: int = None) -> np.ndarray:
    if avg_block_len is None:
        avg_block_len = max(1, int(np.ceil(n ** (1 / 3))))
    p = 1.0 / avg_block_len
    starts = rng.integers(0, n, size=(B, n))
    new_block = rng.random(size=(B, n)) < p
    new_block[:, 0] = True
    indices = np.empty((B, n), dtype=int)
    indices[:, 0] = starts[:, 0]
    for t in range(1, n):
        indices[:, t] = np.where(new_block[:, t],
                                 starts[:, t],
                                 (indices[:, t - 1] + 1) % n)
    return indices


def _bootstrap_ci(forecast_col: str, df: pd.DataFrame, ann: int, gamma: float,
                  B: int = 1000, ci: float = 0.90) -> tuple[float, float]:
    valid = df[["simple_eqprem", "sigma2", forecast_col, "predicted_HM"]].dropna().reset_index(drop=True)
    if len(valid) < 24:
        return np.nan, np.nan
    n = len(valid)
    r = valid["simple_eqprem"].values
    sig2 = valid["sigma2"].values
    f_mod = valid[forecast_col].values
    f_ben = valid["predicted_HM"].values

    w_mod = np.clip(f_mod / (gamma * sig2), 0.0, 1.5)
    w_ben = np.clip(f_ben / (gamma * sig2), 0.0, 1.5)

    rng = np.random.default_rng(42)
    idx = _stationary_bootstrap_indices(n, B, rng)

    def _cer_np(w_arr, r_arr):
        pr = w_arr * r_arr
        return ann * pr.mean(axis=1) - (gamma / 2) * ann * pr.var(axis=1, ddof=1)

    gains_model = _cer_np(w_mod[idx], r[idx])
    gains_bench = _cer_np(w_ben[idx], r[idx])
    gains = (gains_model - gains_bench) * 10_000

    lo = float(np.nanpercentile(gains, (1 - ci) / 2 * 100))
    hi = float(np.nanpercentile(gains, (1 + ci) / 2 * 100))
    return lo, hi


def _breakeven_tc(forecast_col: str, df: pd.DataFrame, ann: int, gamma: float) -> float:
    valid = df[["simple_eqprem", "sigma2", forecast_col, "predicted_HM"]].dropna().reset_index(drop=True)
    if len(valid) < 24:
        return np.nan
    w_model = _weights(forecast_col, valid, gamma)
    w_bench = _weights("predicted_HM", valid, gamma)
    gross_gain = (_cer(w_model, valid["simple_eqprem"], ann, gamma)
                  - _cer(w_bench, valid["simple_eqprem"], ann, gamma)) * 10_000
    mean_turnover_model = w_model.diff().abs().mean()
    mean_turnover_bench = w_bench.diff().abs().mean()
    net_turnover = mean_turnover_model - mean_turnover_bench
    if net_turnover <= 0:
        return np.nan
    return float(gross_gain / (net_turnover * ann))


def _format_predictor_label(col: str, prefix: str) -> str:
    label = col.replace(prefix, "")
    if label == "1N":
        return "1/N combination"
    return label


def compute_cer_gain_table(df, forecasts_out, date_col, oos_start,
                           target_col, var_window, ann, gamma,
                           subperiods, forecast_prefix):
    hist_avg_simple = expanding_historical_avg_simple(df, date_col, oos_start, target_col)
    sigma2 = lagged_rolling_variance(df, target_col, var_window)

    rows = []
    for years, out in forecasts_out.items():
        lookup = pd.DataFrame({date_col: df[date_col],
                               target_col: df[target_col],
                               "sigma2": sigma2,
                               "predicted_HM": hist_avg_simple})
        merged = pd.merge(out, lookup, on=date_col, how="left")
        forecast_cols = [c for c in out.columns if c.startswith(forecast_prefix)]
        for period_label, bounds in subperiods.items():
            lo, hi = bounds
            mask = (merged[date_col] >= lo) & (merged[date_col] <= hi)
            sub = merged[mask].copy()
            for col in forecast_cols:
                valid_n = sub[[target_col, "sigma2", col, "predicted_HM"]].dropna().shape[0]
                if valid_n < 24:
                    continue
                gain = _cer_gain_bps(col, sub, ann, gamma)
                lo90, hi90 = _bootstrap_ci(col, sub, ann, gamma)
                btc = _breakeven_tc(col, sub, ann, gamma)
                rows.append({
                    "Frequency": "Quarterly" if ann == 4 else "Monthly",
                    "Period": period_label,
                    "Window": f"{years}y",
                    "Predictor": _format_predictor_label(col, forecast_prefix),
                    "n": valid_n,
                    "CER gain (bps)": round(gain, 1),
                    "CI lo 90%": round(lo90, 1),
                    "CI hi 90%": round(hi90, 1),
                    "Breakeven TC (bps)": round(btc, 1) if not np.isnan(btc) else np.nan,
                })
    return pd.DataFrame(rows)


def print_cer_summary(cer_df, freq_label):
    for period_label in sorted(cer_df["Period"].unique()):
        tbl = cer_df[cer_df["Period"] == period_label].copy()
        print(f"\n{freq_label} | {period_label}")
        print(tbl[["Window", "Predictor", "n", "CER gain (bps)", "CI lo 90%", "CI hi 90%", "Breakeven TC (bps)"]]
              .to_string(index=False))


# ── Run All Forecasts ─────────────────────────────────────────────────────────

def run_rolling_forecasts(df, predictors, date_col, oos_start, periods_per_year, subperiods):
    """
    For each window size:
      - Compute rolling OOS forecasts for each predictor
      - Compute 1/N combination forecast
      - Compute expanding historical average benchmark
      - Compute OOS R² and Clark-West stat/p-value over each sub-period

    Returns:
      forecasts  : dict[years -> DataFrame]
      r2_tables  : dict[period_label -> DataFrame(rows=predictors+1N, cols=window_years)]
      cw_tables  : dict[period_label -> dict{"stat"|"pval" -> DataFrame}]
    """
    df = df.reset_index(drop=True)
    oos_mask = df[date_col] >= oos_start

    hist_avg_full = expanding_historical_avg(df, date_col, oos_start)
    hist_avg_oos  = hist_avg_full[oos_mask].values

    forecasts_out = {}
    r2_data  = {label: [] for label in subperiods}
    cw_data  = {label: {"stat": [], "pval": []} for label in subperiods}

    for years in WINDOW_YEARS:
        window = years * periods_per_year
        print(f"  Window = {years}y ({window} periods)...")

        out = df.loc[oos_mask, [date_col, TARGET]].copy().reset_index(drop=True)
        out.rename(columns={TARGET: "eqprem_actual"}, inplace=True)
        out["hist_avg_benchmark"] = hist_avg_oos

        hat_cols = []
        for pred in predictors:
            forecasts = rolling_oos_forecast(df, pred, date_col, oos_start, window)
            col = f"y_hat_{pred}"
            out[col] = forecasts[oos_mask].values
            hat_cols.append(col)

        out["y_hat_1N"] = out[hat_cols].mean(axis=1)
        forecasts_out[years] = out

        all_cols   = list(zip(predictors, hat_cols)) + [("1/N_combination", "y_hat_1N")]

        for label, bounds in subperiods.items():
            r2_row   = {"window": f"{years}y"}
            cw_s_row = {"window": f"{years}y"}
            cw_p_row = {"window": f"{years}y"}

            for pred, col in all_cols:
                r2_row[pred]   = oos_r2_subperiod(out, date_col, col, bounds)
                cw_s, cw_p     = cw_subperiod(out, date_col, col, bounds)
                cw_s_row[pred] = cw_s
                cw_p_row[pred] = cw_p

            r2_data[label].append(r2_row)
            cw_data[label]["stat"].append(cw_s_row)
            cw_data[label]["pval"].append(cw_p_row)

    # Build tables
    r2_tables = {}
    cw_tables = {}
    for label in subperiods:
        r2_tables[label] = pd.DataFrame(r2_data[label]).set_index("window").T
        cw_tables[label] = {
            "stat": pd.DataFrame(cw_data[label]["stat"]).set_index("window").T,
            "pval": pd.DataFrame(cw_data[label]["pval"]).set_index("window").T,
        }

    return forecasts_out, r2_tables, cw_tables


# ── Console Summary ───────────────────────────────────────────────────────────

def print_r2_cw_summary(r2_tables, cw_tables, freq_label):
    for period_label in r2_tables:
        r2_table  = r2_tables[period_label]
        cw_stat   = cw_tables[period_label]["stat"]
        cw_pval   = cw_tables[period_label]["pval"]
        sep = "-" * 105
        print(f"\n{'=' * 105}")
        print(f"  OOS R² & Clark-West Test -- {freq_label} | Period: {period_label}")
        print(f"  (benchmark = expanding historical average  |  CW p-value is one-sided)")
        print(f"{'=' * 105}")
        cols = list(r2_table.columns)
        hdr  = f"  {'Predictor':<20}"
        for c in cols:
            hdr += f"  {'R2 ' + c:>10}  {'CW-t ' + c:>10}  {'p ' + c:>8}"
        print(hdr)
        print(sep)
        for pred in r2_table.index:
            row = f"  {pred:<20}"
            for c in cols:
                row += f"  {r2_table.loc[pred, c]:>10.4f}  {cw_stat.loc[pred, c]:>10.4f}  {cw_pval.loc[pred, c]:>8.4f}"
            print(row)
        print(sep)


# ── Excel Export ──────────────────────────────────────────────────────────────

def build_results_sheet(r2_tables, cw_tables, freq_label):
    """
    Build a flat DataFrame with columns:
      Frequency | Predictor | Period | R2_5y | R2_10y | R2_15y |
      CW_stat_5y | CW_stat_10y | CW_stat_15y |
      CW_pval_5y | CW_pval_10y | CW_pval_15y
    """
    rows = []
    for period_label in r2_tables:
        r2  = r2_tables[period_label]
        cws = cw_tables[period_label]["stat"]
        cwp = cw_tables[period_label]["pval"]
        for pred in r2.index:
            entry = {"Frequency": freq_label, "Predictor": pred, "Period": period_label}
            for c in r2.columns:
                entry[f"R2_{c}"]      = r2.loc[pred, c]
                entry[f"CW_stat_{c}"] = cws.loc[pred, c]
                entry[f"CW_pval_{c}"] = cwp.loc[pred, c]
            rows.append(entry)
    return pd.DataFrame(rows)


def save_to_excel(forecasts_q, r2_tables_q, cw_tables_q,
                  forecasts_m, r2_tables_m, cw_tables_m,
                  cer_q, cer_m, output_path):
    """
    Sheets:
      Quarterly_5y / 10y / 15y  : full time-series of forecasts
      Monthly_5y   / 10y / 15y  : same
      Results                   : OOS R² + CW stat + CW p-value for all
      CER_gain                  : CER gains, bootstrap 90% CI, breakeven costs
    """
    results_q  = build_results_sheet(r2_tables_q, cw_tables_q, "Quarterly")
    results_m  = build_results_sheet(r2_tables_m, cw_tables_m, "Monthly")
    results_all = pd.concat([results_q, results_m], ignore_index=True)
    cer_all     = pd.concat([cer_q, cer_m], ignore_index=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for years in WINDOW_YEARS:
            forecasts_q[years].to_excel(writer, sheet_name=f"Quarterly_{years}y", index=False)
        for years in WINDOW_YEARS:
            forecasts_m[years].to_excel(writer, sheet_name=f"Monthly_{years}y", index=False)
        results_all.to_excel(writer, sheet_name="Results", index=False)
        cer_all.to_excel(writer, sheet_name="CER_gain", index=False)

    print(f"Saved: {output_path}")


# ── Plotting ─────────────────────────────────────────────────────────────────

def date_code_to_datetime(codes, freq):
    """
    Convert integer date codes to pandas Timestamps.
      Quarterly : YYYYQ  -> first month of quarter (e.g. 19791 -> 1979-01-01)
      Monthly   : YYYYMM -> first day of month     (e.g. 197901 -> 1979-01-01)
    """
    dates = []
    for c in codes:
        if freq == "quarterly":
            year = int(c) // 10
            quarter = int(c) % 10
            month = (quarter - 1) * 3 + 1
            dates.append(pd.Timestamp(year=year, month=month, day=1))
        else:
            year = int(c) // 100
            month = int(c) % 100
            dates.append(pd.Timestamp(year=year, month=month, day=1))
    return dates


def plot_forecasts(forecasts, date_col, freq, predictors, hat_prefix="y_hat_"):
    """
    For each window size, plot each predictor's OOS forecast vs actual log eqprem.
    Also plots the 1/N combination and the historical average benchmark.
    """
    for years, out in forecasts.items():
        dates = date_code_to_datetime(out[date_col], freq)

        n_preds = len(predictors)
        ncols = 3
        nrows = (n_preds + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5), sharex=False)
        axes = axes.flatten()

        for i, pred in enumerate(predictors):
            ax = axes[i]
            col = f"{hat_prefix}{pred}"
            ax.plot(dates, out["eqprem_actual"].values, color="black", linewidth=1.2,
                    label="Actual log eqprem", zorder=3)
            ax.plot(dates, out["hist_avg_benchmark"].values, color="grey", linewidth=0.9,
                    linestyle="--", label="Hist avg", zorder=2)
            if col in out.columns:
                ax.plot(dates, out[col].values, color="steelblue", linewidth=1.0,
                        label=f"Forecast ({pred})", zorder=4)
            ax.set_title(pred, fontsize=9)
            ax.xaxis.set_major_locator(mdates.YearLocator(10))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="upper left")

        # 1/N combination plot in last used panel
        if len(predictors) < len(axes):
            ax = axes[len(predictors)]
            ax.plot(dates, out["eqprem_actual"].values, color="black", linewidth=1.2,
                    label="Actual log eqprem", zorder=3)
            ax.plot(dates, out["hist_avg_benchmark"].values, color="grey", linewidth=0.9,
                    linestyle="--", label="Hist avg", zorder=2)
            ax.plot(dates, out["y_hat_1N"].values, color="darkorange", linewidth=1.0,
                    label="1/N combination", zorder=4)
            ax.set_title("1/N combination", fontsize=9)
            ax.xaxis.set_major_locator(mdates.YearLocator(10))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="upper left")
            last_used = len(predictors) + 1
        else:
            last_used = len(predictors)

        # Hide any remaining empty axes
        for j in range(last_used, len(axes)):
            axes[j].set_visible(False)

        freq_label = freq.capitalize()
        fig.suptitle(
            f"OLS Rolling OOS Forecasts vs Actual Log Equity Premium\n"
            f"{freq_label} | Window = {years}y",
            fontsize=11, y=1.01
        )
        plt.tight_layout()
        plt.savefig(f"Forecasts_{freq_label}_{years}y.png", dpi=150, bbox_inches="tight")
        print(f"Saved plot: Forecasts_{freq_label}_{years}y.png")
        plt.show()


# ── 1/N Combination Forecast Plot ────────────────────────────────────────────

def plot_1n_forecast(forecasts_q, forecasts_m):
    """
    Single figure: 2 rows (Quarterly, Monthly) x 3 cols (5y, 10y, 15y).
    Each panel: actual log eqprem (black) vs 1/N combination forecast (orange)
    and historical average benchmark (grey dashed).
    """
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=False)

    configs = [
        (forecasts_q, "yyyyq",  "quarterly", "Quarterly", 0),
        (forecasts_m, "yyyymm", "monthly",   "Monthly",   1),
    ]

    for forecasts, date_col, freq, freq_label, row in configs:
        for col_idx, years in enumerate(WINDOW_YEARS):
            ax  = axes[row][col_idx]
            out = forecasts[years]
            dates = date_code_to_datetime(out[date_col], freq)

            ax.plot(dates, out["eqprem_actual"].values,
                    color="black", linewidth=1.2, label="Actual log eqprem", zorder=3)
            ax.plot(dates, out["hist_avg_benchmark"].values,
                    color="grey", linewidth=0.9, linestyle="--", label="Hist avg", zorder=2)
            ax.plot(dates, out["y_hat_1N"].values,
                    color="darkorange", linewidth=1.1, label="1/N forecast", zorder=4)

            ax.set_title(f"{freq_label} | Window = {years}y", fontsize=9)
            ax.xaxis.set_major_locator(mdates.YearLocator(10))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.tick_params(labelsize=7)
            if row == 0 and col_idx == 0:
                ax.legend(fontsize=7, loc="upper left")

    fig.suptitle(
        "1/N Combination OOS Forecast vs Actual Log Equity Premium",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig("Forecasts_1N.png", dpi=150, bbox_inches="tight")
    print("Saved plot: Forecasts_1N.png")
    plt.show()


# ── S_fr: Average Forecast-Return Correlation ────────────────────────────────

def compute_sfr(forecasts, date_col, freq, predictors, hat_prefix="y_hat_",
                sfr_start=19791, sfr_end=20224):
    """
    For each window size and rolling window W, compute:
      S_fr_t(W) = (1/N) * sum_i corr(actual_{t-W+1:t}, forecast_i_{t-W+1:t})
    over all t where date is within [sfr_start, sfr_end].
    Returns dict: years -> DataFrame(date, S_fr_W{periods})
    """
    results = {}
    for years, out in forecasts.items():
        dates = date_code_to_datetime(out[date_col], freq)
        actual = out["eqprem_actual"].values
        pred_cols = [f"{hat_prefix}{p}" for p in predictors if f"{hat_prefix}{p}" in out.columns]
        pred_matrix = out[pred_cols].values
        N = len(pred_cols)
        T = len(actual)

        # Convert sfr_start/sfr_end to datetime for comparison
        if freq == "quarterly":
            start_dt = date_code_to_datetime([sfr_start], freq)[0]
            end_dt   = date_code_to_datetime([sfr_end],   freq)[0]
        else:
            start_dt = date_code_to_datetime([sfr_start], freq)[0]
            end_dt   = date_code_to_datetime([sfr_end],   freq)[0]

        # Rolling window in periods
        W = years * (4 if freq == "quarterly" else 12)

        rows = []
        for t in range(2, T + 1):
            dt = dates[t - 1]
            if dt < start_dt or dt > end_dt:
                continue
            start_w = max(0, t - W)
            r_w = actual[start_w:t]
            P_w = pred_matrix[start_w:t, :]
            if len(r_w) < 2:
                continue
            rhos = []
            for i in range(N):
                p_i = P_w[:, i]
                valid = ~(np.isnan(r_w) | np.isnan(p_i))
                if valid.sum() < 2:
                    continue
                rho = np.corrcoef(r_w[valid], p_i[valid])[0, 1]
                if not np.isnan(rho):
                    rhos.append(rho)
            if rhos:
                rows.append({"date": dt, "S_fr": np.mean(rhos)})

        results[years] = pd.DataFrame(rows)
    return results


def plot_sfr(forecasts_q, forecasts_m):
    """
    Plot S_fr separately for Quarterly and Monthly, saved as two PNG files.
    Forecasts must start from 1965 so rolling windows are fully populated by 1979.
    """
    sfr_q = compute_sfr(forecasts_q, "yyyyq",  "quarterly", PREDICTORS_Q,
                        sfr_start=OOS_START_Q, sfr_end=20224)
    sfr_m = compute_sfr(forecasts_m, "yyyymm", "monthly",   PREDICTORS_M,
                        sfr_start=OOS_START_M, sfr_end=202212)

    colors    = {5: "steelblue",  10: "darkorange", 15: "darkgreen"}
    lws       = {5: 1.4,          10: 1.4,          15: 1.4}
    linestyle = {5: "-",          10: "--",          15: ":"}

    configs = [
        (sfr_q, "Quarterly", "S_fr_Quarterly.png", 4,  "quarters"),
        (sfr_m, "Monthly",   "S_fr_Monthly.png",  12, "months"),
    ]

    for sfr_data, freq_label, fname, ppy, period_unit in configs:
        fig, ax = plt.subplots(figsize=(9, 4))
        for years in WINDOW_YEARS:
            df_sfr = sfr_data[years]
            if df_sfr.empty:
                continue
            periods = years * ppy
            ax.plot(df_sfr["date"], df_sfr["S_fr"],
                    color=colors[years], linewidth=lws[years],
                    linestyle=linestyle[years],
                    label=f"W={periods} {period_unit}")
        ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
        ax.set_xlabel("Date")
        ax.set_ylabel("$S^{fr}$")
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlim(right=pd.Timestamp("2025-01-01"))
        ax.tick_params(labelsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=9, frameon=True)
        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {fname}")
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading data from: {FILE_PATH}")
    df_q, df_m = load_data(FILE_PATH)

    print("\nRunning quarterly rolling-window OOS forecasts...")
    forecasts_q, r2_tables_q, cw_tables_q = run_rolling_forecasts(
        df_q, PREDICTORS_Q, "yyyyq", OOS_START_Q, periods_per_year=4, subperiods=PERIODS_Q
    )
    print_r2_cw_summary(r2_tables_q, cw_tables_q, "Quarterly")

    print("\nRunning monthly rolling-window OOS forecasts...")
    forecasts_m, r2_tables_m, cw_tables_m = run_rolling_forecasts(
        df_m, PREDICTORS_M, "yyyymm", OOS_START_M, periods_per_year=12, subperiods=PERIODS_M
    )
    print_r2_cw_summary(r2_tables_m, cw_tables_m, "Monthly")

    print("\nComputing CER gain for rolling-window forecasts...")
    cer_q = compute_cer_gain_table(
        df_q, forecasts_q, "yyyyq", OOS_START_Q,
        target_col="simple_eqprem", var_window=40, ann=4, gamma=3.0,
        subperiods=PERIODS_Q, forecast_prefix="y_hat_"
    )
    cer_m = compute_cer_gain_table(
        df_m, forecasts_m, "yyyymm", OOS_START_M,
        target_col="simple_eqprem", var_window=120, ann=12, gamma=3.0,
        subperiods=PERIODS_M, forecast_prefix="y_hat_"
    )
    print_cer_summary(cer_q, "Quarterly")
    print_cer_summary(cer_m, "Monthly")

    save_to_excel(
        forecasts_q, r2_tables_q, cw_tables_q,
        forecasts_m, r2_tables_m, cw_tables_m,
        cer_q, cer_m, OUTPUT_PATH
    )

    print("\nGenerating 1/N forecast plot...")
    plot_1n_forecast(forecasts_q, forecasts_m)

    print("\nRunning extended quarterly forecasts from 1965 for S_fr...")
    forecasts_q_ext, _, _ = run_rolling_forecasts(
        df_q, PREDICTORS_Q, "yyyyq", OOS_START_Q_EXT, periods_per_year=4, subperiods=PERIODS_Q
    )
    print("\nRunning extended monthly forecasts from 1965 for S_fr...")
    forecasts_m_ext, _, _ = run_rolling_forecasts(
        df_m, PREDICTORS_M, "yyyymm", OOS_START_M_EXT, periods_per_year=12, subperiods=PERIODS_M
    )
    print("\nGenerating S_fr plot...")
    plot_sfr(forecasts_q_ext, forecasts_m_ext)

    return forecasts_q, r2_tables_q, cw_tables_q, forecasts_m, r2_tables_m, cw_tables_m


if __name__ == "__main__":
    forecasts_q, r2_tables_q, cw_tables_q, forecasts_m, r2_tables_m, cw_tables_m = main()