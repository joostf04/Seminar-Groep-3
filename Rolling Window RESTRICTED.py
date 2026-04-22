"""
OLS Rolling-Window OOS Forecasts WITH Campbell-Thompson (2008) Restrictions

Restrictions applied exactly as in the reference implementation:
  1. Sign restriction: if sign(beta) conflicts with expected sign, set beta = 0
     (forecast collapses to the intercept only for that period).
     NOTE: no positivity floor is applied to the forecast.
  2. Predictor is lagged by 1 period before fitting to avoid look-ahead bias.

Expected signs follow Campbell & Thompson (2008) as in the reference code:
  Positive (+): d/p, d/y, e/p, d/e, svar, b/m, ltr, tms, dfy, dfr
  Negative (-): ntis, tbl, lty, infl, i/k

Also computes:
  - 1/N combination forecast (equal-weighted average across all CT-restricted forecasts)
  - OOS R² over three sub-periods: 1979-2022, 1979-1993, 1994-2022
  - Clark-West (2007) test statistic and one-sided p-value vs historical average benchmark

Window sizes : 5, 10, 15 years
First OOS period : 1979Q1 (quarterly) / 1979M1 (monthly)
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

# ── Configuration ────────────────────────────────────────────────────────────

FILE_PATH   = "Data_Seminar.xlsx"
OUTPUT_PATH = "OLS_Forecasts_CT.xlsx"
TARGET      = "log eqprem"

WINDOW_YEARS = [5, 10, 15]

OOS_START_Q = 19791
OOS_START_M = 197901

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

# Expected signs as in the reference code (Campbell & Thompson 2008).
# Key: exact column name from the DataFrame.
# +1 = positive expected sign for beta, -1 = negative expected sign.
EXPECTED_SIGNS = {
    "d/p":  +1,   # dividend-price ratio
    "d/y":  +1,   # dividend yield
    "e/p":  +1,   # earnings-price ratio
    "d/e":  +1,   # dividend payout ratio  (reference: DE = +1)
    "svar": +1,   # stock variance         (reference: SVAR = +1)
    "b/m":  +1,   # book-to-market ratio
    "ntis": -1,   # net equity issuance    (reference: NTIS = -1)
    "tbl":  -1,   # T-bill rate
    "lty":  -1,   # long-term yield
    "ltr":  +1,   # long-term return       (reference: LTR = +1)
    "tms":  +1,   # term spread
    "dfy":  +1,   # default yield spread   (reference: DFY = +1)
    "dfr":  +1,   # default return spread  (reference: DFR = +1)
    "infl": -1,   # inflation
    "i/k":  -1,   # investment-capital ratio (reference: IK = -1)
}


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
    Expanding-window mean of eqprem using only data strictly before t.
    """
    df = df.reset_index(drop=True)
    oos_idx = df.index[df[date_col] >= oos_start].tolist()
    hist_avg = pd.Series(np.nan, index=df.index)
    for t in oos_idx:
        past = df.loc[:t - 1, TARGET].dropna()
        if len(past) > 0:
            hist_avg.at[t] = past.mean()
    return hist_avg


# ── Campbell-Thompson Restricted Rolling Forecast ─────────────────────────────

def rolling_oos_forecast_ct(df, predictor, date_col, oos_start, window):
    """
    Rolling OOS forecast with Campbell-Thompson (2008) sign restriction.

    At each period t:
      1. Lag the predictor by 1 period (pred_{t-1} predicts eqprem_t).
      2. Fit OLS on the rolling window [t-window, t-1] of complete cases.
      3. Sign restriction: if sign(beta) != expected sign, set beta = 0.
      4. Forecast: yhat_t = alpha + beta_restricted * pred_{t-1}

    No positivity floor is applied (consistent with reference implementation).
    """
    expected_sign = EXPECTED_SIGNS.get(predictor, None)

    df = df.reset_index(drop=True)

    # Build lagged predictor aligned with eqprem
    data = df[[date_col, TARGET, predictor]].copy()
    lag_col = f"{predictor}_lag"
    data[lag_col] = data[predictor].shift(1)
    data = data.dropna(subset=[lag_col]).reset_index(drop=True)

    oos_idx_in_data = data.index[data[date_col] >= oos_start].tolist()
    forecasts = pd.Series(np.nan, index=data.index)

    for t in oos_idx_in_data:
        # Rolling training window: window rows immediately before t
        train = data.iloc[max(0, t - window): t][[lag_col, TARGET]].dropna()

        if len(train) < 3:
            continue

        X_train = sm.add_constant(train[lag_col].values, has_constant="add")
        y_train = train[TARGET].values

        try:
            model = sm.OLS(y_train, X_train).fit()
        except Exception:
            continue

        alpha = float(model.params[0])
        beta  = float(model.params[1])

        # Sign restriction: zero out beta if it has the wrong sign
        if expected_sign is not None:
            if (beta > 0 and expected_sign < 0) or (beta < 0 and expected_sign > 0):
                beta = 0.0

        x_t = data.loc[t, lag_col]
        if pd.isna(x_t):
            continue

        forecasts.at[t] = alpha + beta * x_t

    # Re-index back onto the original df index via date alignment
    date_to_forecast = dict(zip(data[date_col], forecasts))
    result = df[date_col].map(date_to_forecast)
    result.index = df.index
    return result


# ── OOS R² ────────────────────────────────────────────────────────────────────

def oos_r2(actual, forecast, benchmark):
    valid = actual.notna() & forecast.notna() & benchmark.notna()
    ss_model = ((actual[valid] - forecast[valid]) ** 2).sum()
    ss_bench = ((actual[valid] - benchmark[valid]) ** 2).sum()
    if ss_bench == 0:
        return np.nan
    return 1 - ss_model / ss_bench


def oos_r2_subperiod(out_df, date_col, forecast_col, period_bounds):
    lo, hi = period_bounds
    mask = (out_df[date_col] >= lo) & (out_df[date_col] <= hi)
    sub  = out_df[mask]
    return oos_r2(sub["eqprem_actual"], sub[forecast_col], sub["hist_avg_benchmark"])


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
    lo, hi = period_bounds
    mask = (out_df[date_col] >= lo) & (out_df[date_col] <= hi)
    sub  = out_df[mask]
    return clark_west(sub["eqprem_actual"], sub[forecast_col], sub["hist_avg_benchmark"])


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

def run_rolling_forecasts_ct(df, predictors, date_col, oos_start, periods_per_year, subperiods):
    """
    CT-restricted rolling OOS forecasts for all window sizes and predictors.

    Returns:
      forecasts  : dict[years -> DataFrame]
      r2_tables  : dict[period_label -> DataFrame]
      cw_tables  : dict[period_label -> dict{stat, pval -> DataFrame}]
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
            forecasts = rolling_oos_forecast_ct(df, pred, date_col, oos_start, window)
            col = f"y_hat_ct_{pred}"
            out[col] = forecasts[oos_mask].values
            hat_cols.append(col)

        # 1/N combination of CT-restricted forecasts
        out["y_hat_ct_1N"] = out[hat_cols].mean(axis=1)
        forecasts_out[years] = out

        all_cols = list(zip(predictors, hat_cols)) + [("1/N_combination", "y_hat_ct_1N")]

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
        r2_table = r2_tables[period_label]
        cw_stat  = cw_tables[period_label]["stat"]
        cw_pval  = cw_tables[period_label]["pval"]
        sep = "-" * 105
        print(f"\n{'=' * 105}")
        print(f"  CT-Restricted OOS R² & Clark-West -- {freq_label} | Period: {period_label}")
        print(f"  (Sign restriction only; no positivity floor; predictor lagged 1 period)")
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
    results_q   = build_results_sheet(r2_tables_q, cw_tables_q, "Quarterly")
    results_m   = build_results_sheet(r2_tables_m, cw_tables_m, "Monthly")
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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading data from: {FILE_PATH}")
    df_q, df_m = load_data(FILE_PATH)

    print("\nExpected signs (matching reference implementation):")
    for pred, sign in EXPECTED_SIGNS.items():
        print(f"  {pred:<8}: {'positive (+1)' if sign == 1 else 'negative (-1)'}")

    print("\nRunning quarterly CT-restricted rolling-window OOS forecasts...")
    forecasts_q, r2_tables_q, cw_tables_q = run_rolling_forecasts_ct(
        df_q, PREDICTORS_Q, "yyyyq", OOS_START_Q, periods_per_year=4, subperiods=PERIODS_Q
    )
    print_r2_cw_summary(r2_tables_q, cw_tables_q, "Quarterly")

    print("\nRunning monthly CT-restricted rolling-window OOS forecasts...")
    forecasts_m, r2_tables_m, cw_tables_m = run_rolling_forecasts_ct(
        df_m, PREDICTORS_M, "yyyymm", OOS_START_M, periods_per_year=12, subperiods=PERIODS_M
    )
    print_r2_cw_summary(r2_tables_m, cw_tables_m, "Monthly")

    print("\nComputing CER gain for CT-restricted rolling-window forecasts...")
    cer_q = compute_cer_gain_table(
        df_q, forecasts_q, "yyyyq", OOS_START_Q,
        target_col="simple_eqprem", var_window=40, ann=4, gamma=3.0,
        subperiods=PERIODS_Q, forecast_prefix="y_hat_ct_"
    )
    cer_m = compute_cer_gain_table(
        df_m, forecasts_m, "yyyymm", OOS_START_M,
        target_col="simple_eqprem", var_window=120, ann=12, gamma=3.0,
        subperiods=PERIODS_M, forecast_prefix="y_hat_ct_"
    )
    print_cer_summary(cer_q, "Quarterly")
    print_cer_summary(cer_m, "Monthly")

    save_to_excel(
        forecasts_q, r2_tables_q, cw_tables_q,
        forecasts_m, r2_tables_m, cw_tables_m,
        cer_q, cer_m, OUTPUT_PATH
    )

    return forecasts_q, r2_tables_q, cw_tables_q, forecasts_m, r2_tables_m, cw_tables_m


if __name__ == "__main__":
    forecasts_q, r2_tables_q, cw_tables_q, forecasts_m, r2_tables_m, cw_tables_m = main()