import time
import warnings
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import spearmanr, rankdata

warnings.filterwarnings("ignore", "invalid value", RuntimeWarning)

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_excel("Data_Seminar.xlsx", sheet_name="Monthly")
df = df.drop(columns=["i/k (NOT HERE)"], errors="ignore")
df["date"] = pd.to_datetime(df["yyyymm"].astype(str), format="%Y%m")
# Rename: use "log eqprem" for all predictions/metrics; keep "simple eqprem" for CER only
df.rename(columns={"log eqprem": "eqprem", "simple eqprem": "simple_eqprem"}, inplace=True)
ANN        = 12          # monthly annualisation
WINDOWS    = [60, 120, 180]  # 60, 120, 180 months
VAR_WINDOW = 120         # 120 months = 10 years
_VAR_PLOT_W = [60, 120, 180]

df = df.sort_values("date").reset_index(drop=True)
print(f"Monthly | Annualisation: x{ANN} | Rolling windows: {WINDOWS}")

# ── Define predictors ──────────────────────────────────────────────────────────
PREDICTORS = [c for c in df.columns if c not in ("yyyymm", "date", "eqprem", "simple_eqprem", "RETURNS INCL DIV", "RF")]
print(f"Predictors ({len(PREDICTORS)}): {PREDICTORS}\n")

# ── Expected signs for restrictions ────────────────────────────────────────────
EXPECTED_SIGNS = {
    "DP": 1, "DY": 1, "EP": 1, "DE": 1, "SVAR": 1, "BM": 1, "LTR": 1, "TMS": 1, "DFY": 1, "DFR": 1,
    "NTIS": -1, "TBL": -1, "LTY": -1, "INFL": -1, "IK": -1
}

print("\n=== PREDICTOR NAME MATCHING DIAGNOSTIC ===")
print(f"DataFrame column names: {PREDICTORS}")
print(f"EXPECTED_SIGNS keys:    {list(EXPECTED_SIGNS.keys())}")
unmatched = []
for p in PREDICTORS:
    sign = EXPECTED_SIGNS.get(p, "NO MATCH")
    print(f"  Column {p!r:12s} -> EXPECTED_SIGNS: {sign}")
    if sign == "NO MATCH":
        unmatched.append(p)
if unmatched:
    print(f"\n*** WARNING: {len(unmatched)} predictors have NO expected sign: {unmatched} ***")
    print("*** CT restrictions will NEVER bind for these predictors! ***")
    # Fix: create mapping if needed
    _NAME_MAP = {name.upper().replace("/", ""): name for name in PREDICTORS}
    EXPECTED_SIGNS_MAPPED = {}
    for key, sign in EXPECTED_SIGNS.items():
        actual_name = _NAME_MAP.get(key)
        if actual_name:
            EXPECTED_SIGNS_MAPPED[actual_name] = sign
    EXPECTED_SIGNS = EXPECTED_SIGNS_MAPPED
    print(f"*** Fixed EXPECTED_SIGNS to match column names: {EXPECTED_SIGNS} ***")
print("=" * 50 + "\n")

# ── Split boundary ─────────────────────────────────────────────────────────────
IN_SAMPLE_END   = pd.Timestamp("1978-12-01")   # last in-sample month
OOS_START_PRED  = pd.Timestamp("1979-01-01")   # first month we *predict*
OOS_END_PRED    = pd.Timestamp("2022-12-01")   # last month we *predict*
FORECAST_START  = pd.Timestamp("1965-01-01")   # extended start for S^fr windows
ST_START        = pd.Timestamp("1979-01-01")   # first date reported for S_t
# The first prediction uses data up to 1978-12 (the full in-sample window).
# The row we predict is the row whose date == OOS_START_PRED.

oos_mask  = (df["date"] >= FORECAST_START) & (df["date"] <= OOS_END_PRED)
oos_index = df.index[oos_mask].tolist()         # extended indices (1965+) for rolling windows
oos_mask_eval = (df["date"] >= OOS_START_PRED) & (df["date"] <= OOS_END_PRED)
oos_index_eval = df.index[oos_mask_eval].tolist()  # evaluation indices (1979+)
n_oos     = len(oos_index_eval)
n_insample = (df["date"] <= IN_SAMPLE_END).sum()
print(f"In-sample obs (used for 1st prediction): {n_insample}")
print(f"Extended predictions (for rolling windows): {len(oos_index)}  "
      f"({df.loc[oos_index[0], 'date'].strftime('%Y-%m')} - "
      f"{df.loc[oos_index[-1], 'date'].strftime('%Y-%m')})")
print(f"OOS evaluation period: {n_oos}  "
      f"({df.loc[oos_index_eval[0], 'date'].strftime('%Y-%m')} - "
      f"{df.loc[oos_index_eval[-1], 'date'].strftime('%Y-%m')})\n")

# ── Storage ────────────────────────────────────────────────────────────────────
results = {}   # predictor -> DataFrame with columns: date, actual, predicted

total_start = time.time()

####################################################
# Predictions monthly                               #
####################################################

# ── Main loop ──────────────────────────────────────────────────────────────────
for pred in PREDICTORS:
    pred_start = time.time()
    print(f"[{pred}] Starting OLS expanding window ...", flush=True)

    # Drop rows where predictor or target is NaN, then lag predictor by one month.
    # Since pred_t and eqprem_t are dated the same month, pred_{t-1} must be used
    # to predict eqprem_t (the predictor is not yet known at the start of month t).
    data = df[["date", "eqprem", pred]].dropna().copy()
    lag_col = f"{pred}_lag"
    data[lag_col] = data[pred].shift(1)          # pred_{t-1} aligned with eqprem_t
    data = data.dropna(subset=[lag_col])         # drop first row (no lag available)

    actuals      = []
    predicted    = []
    dates        = []

    for oos_row in oos_index:
        # Expanding window: use all rows with date < date_of_oos_row
        oos_date    = df.loc[oos_row, "date"]
        train_mask  = data["date"] < oos_date
        train       = data[train_mask]

        if len(train) < 5:          # guard: not enough data
            continue

        # Train: eqprem_t ~ pred_{t-1}  (no look-ahead)
        X_train = sm.add_constant(train[lag_col].values, has_constant="add")
        y_train = train["eqprem"].values

        # Fit OLS
        model = sm.OLS(y_train, X_train).fit()

        # Predict using pred_{t-1}, i.e. the lagged predictor for oos_date
        row_data = data[data["date"] == oos_date]
        if row_data.empty:
            continue

        alpha = float(model.params[0])
        beta  = float(model.params[1])
        expected = EXPECTED_SIGNS.get(pred, 0)
        beta_restricted = beta
        if (beta > 0 and expected < 0) or (beta < 0 and expected > 0):
            beta_restricted = 0.0

        x_val = float(row_data[lag_col].values[0])
        y_hat = alpha + beta_restricted * x_val
        y_hat = max(0.0, y_hat)  # non-negative constraint

        if beta_restricted != beta:
            y_hat_unrestricted = alpha + beta * x_val
            assert not np.allclose(y_hat, y_hat_unrestricted), (
                f"Period {oos_date.strftime('%Y-%m')}, predictor {pred}: "
                "restricted and unrestricted forecasts are identical!"
            )

        dates.append(oos_date)
        actuals.append(row_data["eqprem"].values[0])
        predicted.append(y_hat)

    elapsed = time.time() - pred_start
    oos_df  = pd.DataFrame({"date": dates, "actual": actuals,
                             "predicted": predicted})
    results[pred] = oos_df

    print(f"  Done in {elapsed:.1f}s")

total_elapsed = time.time() - total_start
print(f"\nAll predictors done in {total_elapsed:.1f}s total.")

##################################################
# 1/N combination forecast
##################################################

# Align all individual forecasts on date, then average with equal (1/N) weights.
print("\nComputing 1/N combination forecast ...")

pred_frames = []
for pred, oos_df in results.items():
    pred_frames.append(
        oos_df[["date", "predicted"]].rename(columns={"predicted": pred})
    )

# Merge all forecasts on date (outer join keeps all dates)
from functools import reduce
combined = reduce(lambda a, b: pd.merge(a, b, on="date", how="outer"), pred_frames)
combined = combined.sort_values("date").reset_index(drop=True)

# 1/N weight: simple row-wise mean across whichever predictors are available
forecast_cols = list(results.keys())
combined["predicted_1N"] = combined[forecast_cols].mean(axis=1)

# Add 20 bps cost deduction for economic evaluation
cost = 0.002  # 20 basis points
combined["predicted_1N_cost"] = combined["predicted_1N"] - cost

# Attach actual eqprem
actuals_df = df[["date", "eqprem"]].copy()
combined   = pd.merge(combined, actuals_df, on="date", how="left")

# Metrics for 1/N — evaluation period only (1979+)
eval_mask_1n = combined["date"] >= OOS_START_PRED
errs_1n  = combined.loc[eval_mask_1n, "eqprem"] - combined.loc[eval_mask_1n, "predicted_1N"]
msfe_1n  = (errs_1n ** 2).mean()

# Benchmark MSFE (prevailing mean) — evaluation period only (1979+)
hist_means_1n, hist_actuals_1n = [], []
for oos_row in oos_index_eval:
    oos_date   = df.loc[oos_row, "date"]
    train_mask = df["date"] < oos_date
    if train_mask.sum() < 5:
        continue
    hist_means_1n.append(df.loc[train_mask, "eqprem"].mean())
    hist_actuals_1n.append(df.loc[oos_row, "eqprem"])
bm_msfe_1n = np.mean((np.array(hist_actuals_1n) - np.array(hist_means_1n)) ** 2)
oos_r2_1n  = 1 - msfe_1n / bm_msfe_1n

# Metrics with cost — evaluation period only (1979+)
errs_1n_cost  = combined.loc[eval_mask_1n, "eqprem"] - combined.loc[eval_mask_1n, "predicted_1N_cost"]
msfe_1n_cost  = (errs_1n_cost ** 2).mean()
oos_r2_1n_cost  = 1 - msfe_1n_cost / bm_msfe_1n

print(f"  1/N MSFE: {msfe_1n:.6f} | OOS R²: {oos_r2_1n:.4f}")
print(f"  With 20bps cost: MSFE: {msfe_1n_cost:.6f} | OOS R²: {oos_r2_1n_cost:.4f}")

# Prevailing mean (expanding window) — used later in Decomposition as E_r
# Built aligned to combined["date"] to guarantee date-by-date correspondence.
hist_means_1n_map = {}
for oos_row in oos_index:
    oos_date   = df.loc[oos_row, "date"]
    train_mask = df["date"] < oos_date
    if train_mask.sum() < 5:
        continue
    hist_means_1n_map[oos_date] = df.loc[train_mask, "eqprem"].mean()

n_preds_used = combined[forecast_cols].notna().sum(axis=1).mean()
print(f"  OOS obs: {len(combined)} | Avg predictors per month: {n_preds_used:.1f}")

# ── Save results ───────────────────────────────────────────────────────────────
out_path2 = "ols_oos_predictions_monthly_restricted.xlsx"
with pd.ExcelWriter(out_path2, engine="openpyxl") as writer:
    # --- Summary sheet: one row per predictor + 1/N row ---
    summary_rows = []
    for pred, oos_df in results.items():
        summary_rows.append({"predictor": pred, "n_oos": len(oos_df)})
    summary_rows.append({"predictor": "1/N combination", "n_oos": len(combined)})
    summary_rows.append({"predictor": "1/N combination with 20bps cost", "n_oos": len(combined)})
    pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)

    # --- Predictions sheet: restricted ---
    preds_sheet = combined[["date", "eqprem"]].rename(columns={"eqprem": "actual"}).copy()
    for pred in forecast_cols:
        preds_sheet[f"pred_{pred}"] = combined[pred]
    preds_sheet["predicted_1N"] = combined["predicted_1N"]
    preds_sheet["predicted_1N_cost"] = combined["predicted_1N_cost"]
    preds_sheet.to_excel(writer, sheet_name="Predictions", index=False)

print(f"Results saved to: {out_path2}")

# ── Prevailing mean benchmark DataFrame ────────────────────────────────────────
hm_df = pd.DataFrame({
    "date": list(hist_means_1n_map.keys()),
    "predicted_HM": list(hist_means_1n_map.values())
}).sort_values("date").reset_index(drop=True)

##################################################
# Decomposition                                  #
##################################################

print("\nComputing expanding-window decomposition ...")

# Inputs (T x N matrices, aligned by date)
actual_vec  = combined["eqprem"].values           # (T,)
pred_matrix = combined[forecast_cols].values      # (T, N)
# Map prevailing mean onto combined dates to guarantee alignment
E_r         = np.array([hist_means_1n_map.get(d, np.nan) for d in combined["date"]])
dates_oos   = combined["date"].values
N           = len(forecast_cols)
T           = len(actual_vec)
iota        = np.ones((N, 1))

# Build column names for the flattened N×N matrices: "row_pred|col_pred"
mat_cols = [f"{r}|{c}" for r in forecast_cols for c in forecast_cols]

rows_Sigma      = []
rows_Irred      = []
rows_FCRetCov   = []
rows_VarRhat    = []

for t in range(1, T + 1):
    r  = actual_vec[:t]      # (t,)
    P  = pred_matrix[:t, :]  # (t, N)
    Er = E_r[:t]             # (t,)
    date = dates_oos[t - 1]

    e = r[:, None] - P       # (t, N) forecast errors, NOT demeaned

    Sigma_e     = (e.T @ e) / t
    irreducible = np.mean(r ** 2) * (iota @ iota.T)
    cov_vec     = ((Er @ P) / t).reshape(N, 1)
    fc_ret_cov  = iota @ cov_vec.T + cov_vec @ iota.T
    Var_rhat    = (P.T @ P) / t

    rows_Sigma.append(    [date] + Sigma_e.flatten().tolist())
    rows_Irred.append(    [date] + irreducible.flatten().tolist())
    rows_FCRetCov.append( [date] + fc_ret_cov.flatten().tolist())
    rows_VarRhat.append(  [date] + Var_rhat.flatten().tolist())

cols = ["date"] + mat_cols

with pd.ExcelWriter(out_path2, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    pd.DataFrame(rows_Sigma,    columns=cols).to_excel(writer, sheet_name="Decomp_Sigma_e",    index=False)
    pd.DataFrame(rows_Irred,    columns=cols).to_excel(writer, sheet_name="Decomp_Irreducible", index=False)
    pd.DataFrame(rows_FCRetCov, columns=cols).to_excel(writer, sheet_name="Decomp_FCRetCov",   index=False)
    pd.DataFrame(rows_VarRhat,  columns=cols).to_excel(writer, sheet_name="Decomp_VarRhat",    index=False)

print(f"Decomposition saved to: {out_path2}  ({T} time points, {N}x{N} matrices)")

##################################################
# S^fr and |S_t| — rolling windows             #
##################################################

# Rolling-window correlations (W = 10, 20, 40, 60 quarters):
#   rho_i^t(W)  = corr(r_{t-W+1:t}, r_hat_i_{t-W+1:t})   — N-vector
#   S^fr_t(W)   = (1/N) * sum_i   rho_i^t(W)              — signed average
#   |S_t|(W)    = (1/N) * sum_i | rho_i^t(W) |            — absolute average

# WINDOWS = [10, 20, 40, 60] quarters ≈ 2.5, 5, 10, 15 years

print("\nComputing rolling-window S^fr and |S_t| ...")

sfr_results  = {}   # W -> DataFrame(date, S_fr)
sabs_results = {}   # W -> DataFrame(date, S_abs)
sp_results   = {}   # W -> DataFrame(date, S_p)
rho_rows = {W: [] for W in WINDOWS}  # W -> list of {date, pred: rho}
# Named aliases for the three largest windows (used in rho plot)
rho_rows_180m = rho_rows[WINDOWS[-1]]   # 180 months = 15 years
rho_rows_120m = rho_rows[WINDOWS[-2]]   # 120 months = 10 years
rho_rows_60m = rho_rows[WINDOWS[-3]]   # 60 months =  5 years

for W in WINDOWS:
    print(f"  Window = {W} months ...")
    rows_sfr  = []
    rows_sabs = []
    rows_sp   = []

    for t in range(2, T + 1):          # need >= 2 obs for correlation
        if t < W:
            continue
        r_window   = actual_vec[t - W:t]     # (W,)
        P_window   = pred_matrix[t - W:t, :] # (W, N)
        date       = dates_oos[t - 1]

        # Rolling correlations (Spearman): corr(r, r_hat_i) for each i
        rhos = []
        for i in range(N):
            if np.std(P_window[:, i]) > 0:
                rho, _ = spearmanr(r_window, P_window[:, i])
            else:
                rho = np.nan
            rhos.append(rho)

        # S^fr: signed average correlation
        sfr = np.nanmean(rhos)
        # |S_t|: absolute average correlation
        sabs = np.nanmean(np.abs(rhos))

        # S^p: average pairwise Spearman correlation among forecasts (eq. 6)
        P_ranked = np.apply_along_axis(rankdata, 0, P_window)  # rank each column
        corr_mat = np.corrcoef(P_ranked.T)  # Pearson on ranks = Spearman
        upper_idx = np.triu_indices(N, k=1)
        sp = np.nanmean(corr_mat[upper_idx])

        rows_sfr.append({"date": date, "S_fr": sfr})
        rows_sabs.append({"date": date, "S_abs": sabs})
        rows_sp.append({"date": date, "S_p": sp})

        # Store individual rhos for rho plot (only for largest 3 windows)
        if W in [60, 120, 180]:
            rho_dict = {"date": date}
            for i, pred in enumerate(forecast_cols):
                rho_dict[pred] = rhos[i]
            rho_rows[W].append(rho_dict)

    sfr_results[W]  = pd.DataFrame(rows_sfr)
    sabs_results[W] = pd.DataFrame(rows_sabs)
    sp_results[W]   = pd.DataFrame(rows_sp)

# ── Save S^fr and |S_t| ─────────────────────────────────────────────────────────
with pd.ExcelWriter(out_path2, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    for W in WINDOWS:
        sfr_results[W].to_excel(writer, sheet_name=f"S_fr_W{W}", index=False)
        sabs_results[W].to_excel(writer, sheet_name=f"S_abs_W{W}", index=False)
        sp_results[W].to_excel(writer, sheet_name=f"S_p_W{W}", index=False)

print(f"S^fr and |S_t| saved to: {out_path2}")

# ── Rolling correlations (rho plot) ────────────────────────────────────────────
print("\nSaving rolling correlations for rho plot ...")

with pd.ExcelWriter(out_path2, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    pd.DataFrame(rho_rows_60m).to_excel(writer, sheet_name="Rho_60m", index=False)
    pd.DataFrame(rho_rows_120m).to_excel(writer, sheet_name="Rho_120m", index=False)
    pd.DataFrame(rho_rows_180m).to_excel(writer, sheet_name="Rho_180m", index=False)

print(f"Rolling correlations saved to: {out_path2}")

##################################################
# Correlation-weighted forecasts                #
##################################################

print("\nComputing correlation-weighted forecasts ...")

# Correlation-weighted (CW) forecasts: weight by rolling correlation with eqprem.
#   w_i^t(W) = rho_i^t(W) / sum_j |rho_j^t(W)|
#   r_hat_CW^t(W) = sum_i w_i^t(W) * r_hat_i^t
# where rho_i^t(W) = corr(r_{t-W+1:t}, r_hat_i_{t-W+1:t})

cw_results = {}   # W -> DataFrame(date, predicted_CW_W{W})

for W in _VAR_PLOT_W:
    print(f"  Window = {W} months ...")
    cw_preds = []

    for t in range(2, T + 1):
        if t < W:
            continue
        r_window   = actual_vec[t - W:t]
        P_window   = pred_matrix[t - W:t, :]
        date       = dates_oos[t - 1]

        # Rolling correlations (Spearman)
        rhos = []
        for i in range(N):
            if np.std(P_window[:, i]) > 0:
                rho, _ = spearmanr(r_window, P_window[:, i])
            else:
                rho = 0.0
            rhos.append(rho)

        # Weights: rho_i / sum |rho_j|
        abs_rhos_sum = np.sum(np.abs(rhos))
        if abs_rhos_sum > 0:
            weights = np.array(rhos) / abs_rhos_sum
        else:
            weights = np.ones(N) / N

        # CW forecast: weighted sum of individual forecasts
        P_t = pred_matrix[t - 1, :]  # (N,) forecasts for this t
        cw_pred = np.sum(weights * P_t)
        cw_preds.append({"date": date, "predicted_CW_W{W}": cw_pred})

    cw_results[W] = pd.DataFrame(cw_preds)

# ── Save CW forecasts ──────────────────────────────────────────────────────────
with pd.ExcelWriter(out_path2, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    for W in _VAR_PLOT_W:
        cw_results[W].to_excel(writer, sheet_name=f"CorrWeighted_W{W}", index=False)

print(f"Correlation-weighted forecasts saved in: {out_path2}")

##################################################
# Scaled PCA (sPCA) forecast                    #
##################################################
# Huang, Jiang, Li, Tong & Zhou (2022, Management Science).
#
# At each OOS month t, using only data strictly before t:
#
#   Step 1 — Scale each predictor x_i by its correlation with r:
#             x̃_i = corr(r, x_i) * x_i
#
#   Step 2 — Apply PCA to the scaled predictor matrix X̃.
#             Retain the first K principal components (factors).
#
#   Step 3 — Regress r on the K factors (OLS, expanding window).
#             Forecast r_{t+1} using the factor values at month t.
#
#   Non-negative restriction: if forecast < 0 → set to 0.
#
# Run for K = 1, 2, 3, 4.  Also extract the full-sample idiosyncratic
# covariance matrix (residual covariance after removing K factors)
# and visualise it as a heatmap in the terminal.

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ALL_K = [1, 2, 3, 4]

print("\nComputing sPCA expanding-window forecasts (K = 1–4) ...")

# Build a lagged predictor matrix aligned with eqprem
lag_data = df[["date", "eqprem"] + forecast_cols].copy()
for col in forecast_cols:
    lag_data[col] = lag_data[col].shift(1)
lag_data = lag_data.dropna().reset_index(drop=True)

spca_results     = {}   # K -> DataFrame (sPCA using factors 1..K, nested)

MAX_K  = max(ALL_K)
N_pred = len(forecast_cols)

# One list per K to accumulate results
_dates  = {K: [] for K in ALL_K}
_actual = {K: [] for K in ALL_K}
_pred   = {K: [] for K in ALL_K}

# Best-AIC accumulators (select K with lowest accumulated OOS AIC at each t)
_bestaic_dates  = []
_bestaic_actual = []
_bestaic_pred   = []
_aic_weight_log = []   # store selected K per OOS period for diagnostics
_oos_ss = {K: 0.0 for K in ALL_K}   # accumulated OOS sum of squared errors
_oos_n  = 0                          # number of OOS obs so far


print(f"  Fitting PCA({MAX_K}) once per OOS month; nested factor sets 1..K ...")

for oos_row in oos_index:
    oos_date   = df.loc[oos_row, "date"]
    if oos_date < OOS_START_PRED:
        continue
    train_mask = lag_data["date"] < oos_date
    train      = lag_data[train_mask]

    if len(train) < MAX_K + 2:
        continue

    X_tr = train[forecast_cols].values
    r_tr = train["eqprem"].values

    # Step 1: scale by in-sample OLS slopes β̂_i and apply CT restrictions
    unrestricted_slopes = np.array([
        float(sm.OLS(r_tr, sm.add_constant(X_tr[:, i], has_constant="add")).fit().params[1])
        if X_tr[:, i].std() > 0 else 0.0
        for i in range(X_tr.shape[1])
    ])
    restricted_slopes = np.array([
        slope if not ((slope > 0 and EXPECTED_SIGNS.get(pred, 0) < 0)
                      or (slope < 0 and EXPECTED_SIGNS.get(pred, 0) > 0))
        else 0.0
        for slope, pred in zip(unrestricted_slopes, forecast_cols)
    ])
    n_restricted = np.sum(restricted_slopes == 0.0)
    if n_restricted > 0:
        print(f"Period {oos_date.strftime('%Y-%m')}: {n_restricted}/{len(forecast_cols)} slopes restricted to zero")

    X_tr_scaled = X_tr * restricted_slopes

    # Step 2: standardise → PCA with MAX_K components (fit once)
    scaler   = StandardScaler()
    X_tr_std = scaler.fit_transform(X_tr_scaled)
    zero_std = scaler.scale_ == 0
    if np.any(zero_std):
        X_tr_std[:, zero_std] = 0.0

    pca      = PCA(n_components=MAX_K)
    F_tr     = pca.fit_transform(X_tr_std)       # (T, MAX_K)

    # OOS predictor point
    test_row = lag_data[lag_data["date"] == oos_date]
    if test_row.empty:
        continue
    X_oos     = test_row[forecast_cols].values
    X_oos_std = scaler.transform(X_oos * restricted_slopes)
    if np.any(zero_std):
        X_oos_std[:, zero_std] = 0.0
    F_oos     = pca.transform(X_oos_std)         # (1, MAX_K)

    if n_restricted > 0:
        X_tr_scaled_unrestricted = X_tr * unrestricted_slopes
        scaler_u = StandardScaler()
        X_tr_std_u = scaler_u.fit_transform(X_tr_scaled_unrestricted)
        zero_std_u = scaler_u.scale_ == 0
        if np.any(zero_std_u):
            X_tr_std_u[:, zero_std_u] = 0.0
        pca_u = PCA(n_components=MAX_K)
        F_tr_u = pca_u.fit_transform(X_tr_std_u)
        assert not np.allclose(restricted_slopes, unrestricted_slopes), (
            f"Period {oos_date.strftime('%Y-%m')}: restricted and unrestricted scaling weights are identical!"
        )
        assert not np.allclose(F_tr, F_tr_u), (
            f"Period {oos_date.strftime('%Y-%m')}: restricted and unrestricted sPCA factors are identical!"
        )

    actually_restricted = np.sum(restricted_slopes != unrestricted_slopes)
    if actually_restricted > 0:
        which = [forecast_cols[i] for i in range(len(forecast_cols)) 
                 if restricted_slopes[i] != unrestricted_slopes[i]]
        print(f"  {oos_date.strftime('%Y-%m')}: {actually_restricted} slopes restricted: {which}")

    actual_val = float(test_row["eqprem"].values[0])

    _aic_at_t = {}
    for K in ALL_K:
        # ── sPCA: OLS r ~ F1..FK  (nested cumulative factors) ────────────────
        F_tr_k    = F_tr[:, :K]                                   # (T, K)
        X_reg     = sm.add_constant(F_tr_k, has_constant="add")
        ols_k     = sm.OLS(r_tr, X_reg).fit()
        _aic_at_t[K] = ols_k.aic
        x_oos_reg = np.concatenate([[1.0], F_oos[0, :K]])
        y_hat     = float(ols_k.predict(x_oos_reg)[0])
        if y_hat < 0:
            y_hat = 0.0

        _dates[K].append(oos_date)
        _actual[K].append(actual_val)
        _pred[K].append(y_hat)

    # ── Best-AIC: pick K with lowest accumulated OOS AIC ─────────────
    # Update accumulated OOS squared errors for each K
    for K in ALL_K:
        _oos_ss[K] += (actual_val - _pred[K][-1]) ** 2
    _oos_n += 1

    # Compute accumulated OOS AIC: n*ln(RSS/n) + 2*(K+1)
    if _oos_n >= max(ALL_K) + 2:
        _oos_aic = {K: _oos_n * np.log(_oos_ss[K] / _oos_n) + 2 * (K + 1)
                    for K in ALL_K}
        best_K = min(_oos_aic, key=_oos_aic.get)
    else:
        best_K = 1  # default until enough OOS obs

    y_hat_best = _pred[best_K][-1]
    _bestaic_dates.append(oos_date)
    _bestaic_actual.append(actual_val)
    _bestaic_pred.append(y_hat_best)
    _aic_weight_log.append(best_K)


for K in ALL_K:
    spca_results[K] = pd.DataFrame({
        "date":                 _dates[K],
        "actual":               _actual[K],
        f"predicted_spca_K{K}": _pred[K],
    })
    print(f"  K={K}: {len(_dates[K])} sPCA forecasts"
          f"  ({_dates[K][0].strftime('%Y-%m')} – {_dates[K][-1].strftime('%Y-%m')})")

spca_bestaic_df = pd.DataFrame({
    "date":                    _bestaic_dates,
    "actual":                  _bestaic_actual,
    "predicted_spca_bestAIC":  _bestaic_pred,
})
print(f"  Best-AIC: {len(_bestaic_dates)} forecasts")


# ── AIC selection diagnostics ─────────────────────────────────────────────────
if _aic_weight_log:
    _sel_count = {K: sum(1 for k in _aic_weight_log if k == K) for K in ALL_K}
    print("\n  Best-AIC K selection diagnostics (across all OOS periods):")
    print(f"  {'K':>3s}  {'# times selected':>16s}  {'% selected':>11s}")
    _n_total = len(_aic_weight_log)
    for K in ALL_K:
        print(f"  {K:3d}  {_sel_count[K]:16d}  {100*_sel_count[K]/_n_total:10.1f}%")


# ── Attach sPCA and MVP forecasts to combined frames ─────────────────────────
for K in ALL_K:
    col_spca = f"predicted_spca_K{K}"
    combined = pd.merge(combined, spca_results[K][["date", col_spca]], on="date", how="left")

# Attach best-AIC forecasts
combined = pd.merge(combined, spca_bestaic_df[["date", "predicted_spca_bestAIC"]], on="date", how="left")


# ── Save ─────────────────────────────────────────────────────────────────────
with pd.ExcelWriter(out_path2, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    for K in ALL_K:
        spca_results[K].to_excel(writer, sheet_name=f"sPCA_K{K}", index=False)

print(f"sPCA predictions and idiosyncratic cov saved in: {out_path2}")

##################################################
# Table 2: AIC and BIC for direct sPCA          #
##################################################
# AIC = n*ln(RSS/n) + 2*k,  BIC = n*ln(RSS/n) + k*ln(n)
# k = K+1 (K factors + intercept)

print("\n" + "=" * 60)
print("Table 2: AIC and BIC for direct sPCA specifications (Monthly)")
print("=" * 60)

_AIC_P1 = pd.Timestamp("1979-01-01")
_AIC_PE = pd.Timestamp("2022-12-01")
_AIC_P2 = pd.Timestamp("1994-01-01")
_AIC_P93 = pd.Timestamp("1993-12-01")

_aic_rows = []
for K in ALL_K:
    df_k = spca_results[K].copy()
    col_k = f"predicted_spca_K{K}"
    for p_label, p_start, p_end in [
        ("1979-1993", _AIC_P1, _AIC_P93),
        ("1979-2022", _AIC_P1, _AIC_PE),
        ("1994-2022", _AIC_P2, _AIC_PE),
    ]:
        sub = df_k[(df_k["date"] >= p_start) & (df_k["date"] <= p_end)].dropna(subset=["actual", col_k])
        n   = len(sub)
        if n < K + 2:
            _aic_rows.append({"Panel": "A: AIC", "K": f"K={K}", "Period": p_label, "Value": np.nan})
            _aic_rows.append({"Panel": "B: BIC", "K": f"K={K}", "Period": p_label, "Value": np.nan})
            continue
        rss  = np.sum((sub["actual"].values - sub[col_k].values) ** 2)
        k_p  = K + 1
        aic  = n * np.log(rss / n) + 2 * k_p
        bic  = n * np.log(rss / n) + k_p * np.log(n)
        _aic_rows.append({"Panel": "A: AIC", "K": f"K={K}", "Period": p_label, "Value": round(aic, 4)})
        _aic_rows.append({"Panel": "B: BIC", "K": f"K={K}", "Period": p_label, "Value": round(bic, 4)})

_aic_df    = pd.DataFrame(_aic_rows)
_aic_pivot = _aic_df.pivot_table(index=["Panel", "K"], columns="Period", values="Value", aggfunc="first")
_aic_pivot = _aic_pivot[["1979-1993", "1979-2022", "1994-2022"]]
_aic_pivot["Delta"] = _aic_pivot["1994-2022"] - _aic_pivot["1979-2022"]
print(_aic_pivot.to_string(float_format=lambda x: f"{x:.4f}"))

# ── Save AIC/BIC ──────────────────────────────────────────────────────────────
with pd.ExcelWriter(out_path2, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    _aic_df.to_excel(writer, sheet_name="AIC_BIC_sPCA", index=False)

print(f"AIC/BIC saved to: {out_path2}")


##################################################
# R²_OS — Campbell & Thompson (2008)            #
##################################################
# Equation (13):
#   R²_OS = 1 - Σ_t (r_{t+1} - r̂_{t+1})² / Σ_t (r_{t+1} - r̄_t)²
#
# r̄_t  : prevailing historical mean (expanding window, Goyal & Welch 2008)
# r̂_{t+1} : OLS or combination forecast
#
# Results are reported over two periods:
#   Full sample  : Q1 1979 – Q4 2022 (1979-01 to 2022-12)
#   Post-1994    : Q1 1994 – Q4 2022 (1994-01 to 2022-12)
#
# Significance is assessed using the Clark & West (2007) one-sided test.

from scipy import stats as scipy_stats

print("\n" + "=" * 60)
print("R²_OS — Campbell & Thompson (2008)  [eq. 13]")
print("=" * 60)

FULL_START   = pd.Timestamp("1979-01-01")
FULL_END     = pd.Timestamp("2022-12-01")
PRE94_END    = pd.Timestamp("1993-12-01")
POST94_START = pd.Timestamp("1994-01-01")
DEC2005_END  = pd.Timestamp("2005-12-01")
OOS79_START  = pd.Timestamp("1979-01-01")

PERIODS = {
    "Full OOS (1979 Q1 – 2022 Q4)":      (OOS79_START,  FULL_END),
    "Full OOS (1979 Q1 – 2005 Q4)":      (OOS79_START,  DEC2005_END),
    "Full (1979 Q1 – 2022 Q4)":          (FULL_START,   FULL_END),
    "Full (1979 Q1 – 2005 Q4)":          (FULL_START,   DEC2005_END),
    "Pre-1994 (1979 Q1 – 1993 Q4)":      (FULL_START,   PRE94_END),
    "Post-1994 (1994 Q1 – 2022 Q4)":     (POST94_START, FULL_END),
    "Post-1994 (1994 Q1 – 2005 Q4)":     (POST94_START, DEC2005_END),
}

# Merge the prevailing-mean benchmark onto the combined predictions frame
eval_df = pd.merge(
    combined,
    hm_df[["date", "predicted_HM"]],
    on="date",
    how="inner",
).sort_values("date").reset_index(drop=True)


def r2_os(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray) -> float:
    """Out-of-sample R² (Campbell & Thompson 2008, eq. 13)."""
    return 1.0 - np.sum((actual - predicted) ** 2) / np.sum((actual - benchmark) ** 2)


def clark_west(actual: np.ndarray, predicted: np.ndarray, benchmark: np.ndarray):
    """
    Clark & West (2007) one-sided test.
    H0: model no better than the historical-mean benchmark.
    H1: R²_OS > 0  (one-sided, reject when t-stat is large).
    Returns (t_statistic, p_value).
    """
    f = (
        (actual - benchmark) ** 2
        - (actual - predicted) ** 2
        + (benchmark - predicted) ** 2
    )
    n       = len(f)
    t_stat  = f.mean() / (f.std(ddof=1) / np.sqrt(n))
    p_value = scipy_stats.norm.sf(t_stat)          # one-sided upper tail (standard normal)
    return float(t_stat), float(p_value)

print("\nComputing R²_OS for individual predictors ...")

r2_rows = []

for period_label, (p_start, p_end) in PERIODS.items():
    mask   = (eval_df["date"] >= p_start) & (eval_df["date"] <= p_end)
    sub    = eval_df[mask].copy()

    for pred in forecast_cols:
        valid = sub[["eqprem", pred, "predicted_HM"]].dropna()
        if len(valid) < 10:
            continue
        a, p, bm = valid["eqprem"].values, valid[pred].values, valid["predicted_HM"].values
        r2      = r2_os(a, p, bm)
        t, pval = clark_west(a, p, bm)

        r2_rows.append({
            "Period":      period_label,
            "Predictor":   pred,
            "n":           len(valid),
            "R2_OS (%)":   round(r2 * 100, 2),
            "CW t-stat":   round(t,    3),
            "CW p-value":  round(pval, 4),
        })

    valid_1n = sub[["eqprem", "predicted_1N", "predicted_HM"]].dropna()
    if len(valid_1n) >= 10:
        a, p, bm = valid_1n["eqprem"].values, valid_1n["predicted_1N"].values, valid_1n["predicted_HM"].values
        r2      = r2_os(a, p, bm)
        t, pval = clark_west(a, p, bm)

        r2_rows.append({
            "Period":      period_label,
            "Predictor":   "1/N",
            "n":           len(valid_1n),
            "R2_OS (%)":   round(r2 * 100, 2),
            "CW t-stat":   round(t,    3),
            "CW p-value":  round(pval, 4),
        })

    # sPCA
    for K in ALL_K:
        col_k = f"predicted_spca_K{K}"
        if col_k not in sub.columns:
            continue
        valid_s = sub[["eqprem", col_k, "predicted_HM"]].dropna()
        if len(valid_s) < 10:
            continue
        a, p, bm = valid_s["eqprem"].values, valid_s[col_k].values, valid_s["predicted_HM"].values
        r2      = r2_os(a, p, bm)
        t, pval = clark_west(a, p, bm)

        r2_rows.append({
            "Period":      period_label,
            "Predictor":   f"sPCA K={K}",
            "n":           len(valid_s),
            "R2_OS (%)":   round(r2 * 100, 2),
            "CW t-stat":   round(t,    3),
            "CW p-value":  round(pval, 4),
        })

    # sPCA best-AIC
    for _ba_col, _ba_label in [("predicted_spca_bestAIC", "sPCA best-AIC")]:
        if _ba_col not in sub.columns:
            continue
        valid_ba = sub[["eqprem", _ba_col, "predicted_HM"]].dropna()
        if len(valid_ba) < 10:
            continue
        a, p, bm = valid_ba["eqprem"].values, valid_ba[_ba_col].values, valid_ba["predicted_HM"].values
        r2      = r2_os(a, p, bm)
        t, pval = clark_west(a, p, bm)
        r2_rows.append({
            "Period":      period_label,
            "Predictor":   _ba_label,
            "n":           len(valid_ba),
            "R2_OS (%)":   round(r2 * 100, 2),
            "CW t-stat":   round(t,    3),
            "CW p-value":  round(pval, 4),
        })

r2_df = pd.DataFrame(r2_rows)

# Print summary table
for period_label in PERIODS:
    tbl = r2_df[r2_df["Period"] == period_label].copy()
    print(f"\n{period_label}")
    print(
        tbl[["Predictor", "n", "R2_OS (%)", "CW t-stat", "CW p-value"]]
        .to_string(index=False)
    )

# ── Save to Excel ──────────────────────────────────────────────────────────────
with pd.ExcelWriter(out_path2, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    r2_df.to_excel(writer, sheet_name="R2_OS", index=False)

print(f"\nR²_OS table saved to sheet 'R2_OS' in: {out_path2}")

##################################################
# CER Gain (Campbell & Thompson 2008, eq. 14)   #
##################################################
# Mean-variance investor with risk aversion γ = 3.
# Portfolio weight in equity at time t:
#   w_t = (1/γ) * r̂_{t+1} / σ̂²_{t+1},  clipped to [0, 1.5]
# where σ̂²_{t+1} is the 40-quarter rolling variance of realised returns,
# lagged by 1 quarter (estimated using only data up to t).
#
# CER of a strategy:
#   v̂ = mean(w_t * r_{t+1}) - (γ/2) * var(w_t * r_{t+1})
# annualised (×12) and expressed in basis points (×10000).
#
# CER gain = v̂_model - v̂_benchmark  (benchmark = historical mean)
#
# 90% bootstrap confidence interval (Politis & Romano 1994 stationary bootstrap,
# approximated here with iid block bootstrap, B=1000 draws).
#
# Breakeven TC: the round-trip transaction cost τ (in bps) such that the net
# CER gain = 0, where net CER = gross CER - τ × mean |Δw_t| × 12 × 10000.

print("\nComputing CER gain (eq. 14) ...")

GAMMA      = 3.0
# VAR_WINDOW = 120 months (10 years)
W_MIN, W_MAX = 0.0, 1.5
N_BOOT     = 1000
CI_LEVEL   = 0.90
# ANN already set by FREQUENCY block above

# Rolling variance of equity premium (lagged 1 so t uses data up to t-1)
# CER uses simple (not log) equity premium for portfolio returns
cer_df = eval_df[["date"]].copy()
cer_df = cer_df.merge(df[["date", "simple_eqprem"]], on="date", how="left")
cer_df["sigma2"] = (
    cer_df["simple_eqprem"]
    .rolling(VAR_WINDOW, min_periods=VAR_WINDOW // 2)
    .var()
    .shift(1)
)
cer_df = cer_df.dropna(subset=["sigma2"]).reset_index(drop=True)

# Merge all forecast columns onto cer_df
all_forecast_model_cols = (
    [f"predicted_{p}" for p in PREDICTORS]
    + ["predicted_1N"]
    + [f"predicted_spca_K{K}" for K in ALL_K]
    + ["predicted_spca_bestAIC"]
    + [f"predicted_CW_W{W}" for W in _VAR_PLOT_W]
    + ["predicted_HM"]
)
for col in all_forecast_model_cols:
    if col in eval_df.columns:
        cer_df = cer_df.merge(eval_df[["date", col]], on="date", how="left")

def _weights(forecast_col: str, df: pd.DataFrame) -> pd.Series:
    """Compute mean-variance equity weights clipped to [W_MIN, W_MAX]."""
    w = df[forecast_col] / (GAMMA * df["sigma2"])
    return w.clip(W_MIN, W_MAX)

def _cer(w: pd.Series, r: pd.Series) -> float:
    """Annualised CER in decimal (not bps yet)."""
    port_ret = w * r
    return float(ANN * port_ret.mean() - (GAMMA / 2) * ANN * port_ret.var(ddof=1))

def _cer_gain_bps(forecast_col: str, df: pd.DataFrame) -> float:
    """CER gain of forecast_col vs historical mean, in annualised basis points."""
    valid = df[["simple_eqprem", "sigma2", forecast_col, "predicted_HM"]].dropna()
    if len(valid) < 24:
        return np.nan
    w_model = _weights(forecast_col, valid)
    w_bench = _weights("predicted_HM", valid)
    return (_cer(w_model, valid["simple_eqprem"]) - _cer(w_bench, valid["simple_eqprem"])) * 10_000

def _stationary_bootstrap_indices(n: int, B: int, rng,
                                   avg_block_len: int = None) -> np.ndarray:
    """Politis & Romano (1994) stationary bootstrap index matrix (B, n).
    Block lengths are geometrically distributed with mean avg_block_len."""
    if avg_block_len is None:
        avg_block_len = max(1, int(np.ceil(n ** (1 / 3))))
    p = 1.0 / avg_block_len
    starts    = rng.integers(0, n, size=(B, n))
    new_block = rng.random(size=(B, n)) < p
    new_block[:, 0] = True
    indices = np.empty((B, n), dtype=int)
    indices[:, 0] = starts[:, 0]
    for t in range(1, n):
        indices[:, t] = np.where(new_block[:, t],
                                 starts[:, t],
                                 (indices[:, t - 1] + 1) % n)
    return indices

def _bootstrap_ci(forecast_col: str, df: pd.DataFrame, B: int = N_BOOT,
                  ci: float = CI_LEVEL) -> tuple:
    """Stationary bootstrap CI (Politis & Romano 1994) for CER gain (in bps)."""
    valid = df[["simple_eqprem", "sigma2", forecast_col, "predicted_HM"]].dropna().reset_index(drop=True)
    if len(valid) < 24:
        return (np.nan, np.nan)
    n = len(valid)
    r      = valid["simple_eqprem"].values
    sig2   = valid["sigma2"].values
    f_mod  = valid[forecast_col].values
    f_ben  = valid["predicted_HM"].values

    w_mod = np.clip(f_mod / (GAMMA * sig2), W_MIN, W_MAX)
    w_ben = np.clip(f_ben / (GAMMA * sig2), W_MIN, W_MAX)

    rng  = np.random.default_rng(42)
    idx  = _stationary_bootstrap_indices(n, B, rng)

    def _cer_np(w_arr, r_arr):
        pr = w_arr * r_arr
        return ANN * pr.mean(axis=1) - (GAMMA / 2) * ANN * pr.var(axis=1, ddof=1)

    gains_model = _cer_np(w_mod[idx], r[idx])
    gains_bench = _cer_np(w_ben[idx], r[idx])
    gains       = (gains_model - gains_bench) * 10_000

    lo = float(np.nanpercentile(gains, (1 - ci) / 2 * 100))
    hi = float(np.nanpercentile(gains, (1 + ci) / 2 * 100))
    return (lo, hi)

def _breakeven_tc(forecast_col: str, df: pd.DataFrame) -> float:
    """Breakeven round-trip TC in bps: gross CER gain / mean |Δw| × 12 × 10000."""
    valid = df[["simple_eqprem", "sigma2", forecast_col, "predicted_HM"]].dropna().reset_index(drop=True)
    if len(valid) < 24:
        return np.nan
    w_model = _weights(forecast_col, valid)
    w_bench = _weights("predicted_HM", valid)
    gross_gain = (_cer(w_model, valid["simple_eqprem"]) - _cer(w_bench, valid["simple_eqprem"])) * 10_000
    mean_turnover_model = w_model.diff().abs().mean()
    mean_turnover_bench = w_bench.diff().abs().mean()
    net_turnover = mean_turnover_model - mean_turnover_bench
    if net_turnover <= 0:
        return np.nan
    # gross_gain = tc * net_turnover * ANN * 10000 → tc (in decimal per unit)
    # express tc in bps: tc_bps = tc_decimal * 10000, but here gross is already in bps
    # so tc_bps = gross_gain / (net_turnover * ANN)
    return float(gross_gain / (net_turnover * ANN))

# All model columns excluding the benchmark itself
model_cols_for_cer = [c for c in all_forecast_model_cols if c != "predicted_HM" and c in cer_df.columns]

cer_rows = []
for period_label, (p_start, p_end) in PERIODS.items():
    mask   = (cer_df["date"] >= p_start) & (cer_df["date"] <= p_end)
    sub    = cer_df[mask].copy()
    for col in model_cols_for_cer:
        if col not in sub.columns:
            continue
        valid_n = sub[["simple_eqprem", "sigma2", col, "predicted_HM"]].dropna().shape[0]
        if valid_n < 24:
            continue
        gain      = _cer_gain_bps(col, sub)
        lo, hi    = _bootstrap_ci(col, sub)
        btc       = _breakeven_tc(col, sub)
        # pretty predictor label
        label = (col
                 .replace("predicted_", "")
                 .replace("_1N", "1/N")
                 .replace("spca_K", "sPCA K=")
                 .replace("CW_W", "CW W="))
        cer_rows.append({
            "Period":           period_label,
            "Model":            label,
            "n":                valid_n,
            "CER gain (bps)":   round(gain, 1),
            f"CI lo {int(CI_LEVEL*100)}%": round(lo, 1),
            f"CI hi {int(CI_LEVEL*100)}%": round(hi, 1),
            "Breakeven TC (bps)": round(btc, 1) if not np.isnan(btc) else np.nan,
        })

cer_gain_df = pd.DataFrame(cer_rows)

# Print
for period_label in PERIODS:
    tbl = cer_gain_df[cer_gain_df["Period"] == period_label].copy()
    print(f"\n{period_label}")
    print(tbl.drop(columns="Period").to_string(index=False))

# Save
with pd.ExcelWriter(out_path2, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    cer_gain_df.to_excel(writer, sheet_name="CER_gain", index=False)

print(f"\nCER gain table saved to sheet 'CER_gain' in: {out_path2}")

##################################################
# Plots — S^fr and |S_t| over time              #
##################################################
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

print("\nPlotting S^fr and |S_t| ...")

# Colourblind-safe palette (Wong 2011)
_CB_COLORS = ["#0072B2", "#D55E00", "#009E73"]
_CB_STYLES = ["-", "--", ":"]

# Combined S^fr plot — all windows in one figure
fig, ax = plt.subplots(figsize=(11, 4))
for idx, W in enumerate(WINDOWS):
    _df = sfr_results[W]
    ax.plot(_df["date"], _df["S_fr"],
            color=_CB_COLORS[idx], linestyle=_CB_STYLES[idx],
            linewidth=1.3, label=f"W={W} months")
ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("S$^{fr}$", fontsize=14)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(left=pd.Timestamp("1979-01-01"), right=pd.Timestamp("2025-01-01"))
plt.xticks(rotation=45)
ax.tick_params(axis="both", labelsize=12)
ax.legend(fontsize=13)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("plot_S_fr_combined_monthly_restricted.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: plot_S_fr_combined_monthly_restricted.png")

# Combined S^p plot — all windows in one figure
fig, ax = plt.subplots(figsize=(11, 4))
for idx, W in enumerate(WINDOWS):
    _df = sp_results[W]
    ax.plot(_df["date"], _df["S_p"],
            color=_CB_COLORS[idx], linestyle=_CB_STYLES[idx],
            linewidth=1.3, label=f"W={W} months")
ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("S$^{p}$", fontsize=14)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(left=pd.Timestamp("1979-01-01"), right=pd.Timestamp("2025-01-01"))
plt.xticks(rotation=45)
ax.tick_params(axis="both", labelsize=12)
ax.legend(fontsize=13)
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("plot_S_p_combined_monthly_restricted.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Saved: plot_S_p_combined_monthly_restricted.png")

print("All S^fr / S^p plots saved.")

##################################################
# Plot — Average predictor change vs Eq. Prem   #
# 1994 M1 – 2000 M12                            #
##################################################

print("\nPlotting average predictor change vs equity premium (1994–2000) ...")

_PLOT_START = pd.Timestamp("1994-01-01")
_PLOT_END   = pd.Timestamp("2000-12-01")
_PRE94_END  = pd.Timestamp("1993-12-01")

_pred_plot = df[["date"] + PREDICTORS + ["eqprem"]].copy()
for _pc in PREDICTORS:
    _pred_plot[_pc] = _pred_plot[_pc].shift(1)
_pred_plot = _pred_plot.dropna().sort_values("date").reset_index(drop=True)

_pre94      = _pred_plot[_pred_plot["date"] <= _PRE94_END]
_pred_means = _pre94[PREDICTORS].mean()
_pred_stds  = _pre94[PREDICTORS].std().replace(0, np.nan)

_pred_std_df = _pred_plot.copy()
for _pc in PREDICTORS:
    _pred_std_df[_pc] = (_pred_plot[_pc] - _pred_means[_pc]) / _pred_stds[_pc]

_pred_std_df["avg_pred"] = _pred_std_df[PREDICTORS].mean(axis=1)

_win = (_pred_std_df["date"] >= _PLOT_START) & (_pred_std_df["date"] <= _PLOT_END)
_plot_df = _pred_std_df[_win].copy()

fig, ax1 = plt.subplots(figsize=(11, 4))

color_pred = "#0072B2"   # colourblind-safe blue  (Wong 2011)
color_eq   = "#D55E00"   # colourblind-safe vermillion

_roll = 12  # 12-month rolling average
_plot_df["avg_pred_roll"] = _plot_df["avg_pred"].rolling(_roll, min_periods=1).mean()
_plot_df["eqprem_roll"]   = _plot_df["eqprem"].rolling(_roll, min_periods=1).mean()

ax1.plot(_plot_df["date"], _plot_df["avg_pred"],
         color=color_pred, linewidth=0.8, alpha=0.35, linestyle="-", label="Avg. predictor (raw)")
ax1.plot(_plot_df["date"], _plot_df["avg_pred_roll"],
         color=color_pred, linewidth=2.0, linestyle="-", label="Avg. predictor (12m avg)")
ax1.axhline(0, color=color_pred, linewidth=0.6, linestyle="--", alpha=0.5)
ax1.set_xlabel("Date", fontsize=14)
ax1.set_ylabel("Avg. predictor (z-score, pre-1994 normalised)", fontsize=14, color=color_pred)
ax1.tick_params(axis="both", labelsize=12)
ax1.tick_params(axis="y", labelcolor=color_pred)

ax2 = ax1.twinx()
ax2.plot(_plot_df["date"], _plot_df["eqprem"],
         color=color_eq, linewidth=0.8, alpha=0.35, linestyle="--", dashes=[6, 3], label="Equity premium (raw)")
ax2.plot(_plot_df["date"], _plot_df["eqprem_roll"],
         color=color_eq, linewidth=2.0, alpha=0.9, linestyle="--", dashes=[6, 3], label="Equity premium (12m avg)")
ax2.axhline(0, color=color_eq, linewidth=0.6, linestyle="--", alpha=0.5)
ax2.set_ylabel("Equity premium", fontsize=14, color=color_eq)
ax2.tick_params(axis="y", labelcolor=color_eq)

_lines1, _labs1 = ax1.get_legend_handles_labels()
_lines2, _labs2 = ax2.get_legend_handles_labels()
ax1.legend(_lines1 + _lines2, _labs1 + _labs2, fontsize=9, loc="lower left")

ax1.xaxis.set_major_locator(mdates.YearLocator(1))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.xticks(rotation=45)
ax1.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("plot_pred_vs_eqprem_1994_2000_monthly_restricted.png", dpi=300, bbox_inches="tight")
plt.close(fig)

##################################################
# Discounted MSFE (DMSFE) Combination Weights   #
# Stock & Watson (2004)                         #
##################################################
#
# At each OOS forecast origin t, form exponentially discounted squared-
# error accumulators for each individual predictor i:
#
#   DMSFE_{i,t} = sum_{s=t0}^{t} delta^{t-s} * (r_s - r_hat_{i,s})^2
#
# Weights:  w_{i,t} = DMSFE_{i,t}^{-1} / sum_j DMSFE_{j,t}^{-1}
# Forecast: r_hat_DMSFE_{t+1} = sum_i w_{i,t} * r_hat_{i,t+1}
#
# delta in {0.90, 0.95, 1.0}.  delta=1 recovers undiscounted MSFE weights.
# At t=0 (no prior error history) equal (1/N) weights are used.
#
# Implementation note: the forecast stored at row t uses weights whose
# DMSFE accumulator was last updated at row t-1, i.e. only past errors
# are used — no look-ahead.
#
# Evaluation: same R²_OS, Clark–West, CER gain (+bootstrap CI, breakeven TC)
# as all other methods, over all PERIODS defined above.
#
# Weight diagnostics: cross-sectional weight dispersion, effective N
# (Herfindahl inverse), compared pre- vs post-1994.
#
# DMSFE S^fr: rolling Spearman correlation between the DMSFE combination
# forecast and realised equity premium, for WINDOWS defined above.

print("\n" + "=" * 60)
print("Discounted MSFE Combination Weights  [Stock & Watson 2004]")
print("=" * 60)

DMSFE_DELTAS = [0.95, 0.99, 1.00]

# ── OOS window ────────────────────────────────────────────────────────────
_dm_oos = (
    eval_df[
        (eval_df["date"] >= OOS79_START) & (eval_df["date"] <= FULL_END)
    ]
    .copy()
    .reset_index(drop=True)
)
_T_dm    = len(_dm_oos)
_N_dm    = len(forecast_cols)
_fc_arr  = forecast_cols               # predictor-name columns in eval_df

print(f"\n  OOS window: T={_T_dm} "
      f"({_dm_oos['date'].iloc[0].strftime('%Y-%m')} – "
      f"{_dm_oos['date'].iloc[-1].strftime('%Y-%m')}), "
      f"N={_N_dm} predictors, deltas={DMSFE_DELTAS}")

# Pre-extract arrays (faster inner loop)
_dm_P = _dm_oos[_fc_arr].values.astype(float)   # (T, N) individual forecasts
_dm_r = _dm_oos["eqprem"].values                 # (T,) realised returns

_dmsfe_results = {}   # delta -> {"col", "forecast":(T,), "weights":(T,N)}

for _dl in DMSFE_DELTAS:
    _dmsfe_acc = np.zeros(_N_dm)            # DMSFE_{i, -1} = 0
    _dmsfe_wt  = np.ones(_N_dm) / _N_dm    # equal weights before any errors
    _preds     = np.full(_T_dm, np.nan)
    _wts_ts    = np.full((_T_dm, _N_dm), np.nan)

    for _t in range(_T_dm):

        # ── Step 1: form DMSFE forecast at period t using weights from t-1
        _f_t = _dm_P[_t]                        # (N,) individual forecasts
        _ok  = ~np.isnan(_f_t)
        if _ok.any():
            _w   = np.where(_ok, _dmsfe_wt, 0.0)
            _ws  = _w.sum()
            if _ws > 0:
                _preds[_t] = np.dot(_w / _ws, np.where(_ok, _f_t, 0.0))
        _wts_ts[_t] = _dmsfe_wt.copy()

        # ── Step 2: update DMSFE with the squared error observed at period t
        _r_t = _dm_r[_t]
        if not np.isnan(_r_t):
            _e2  = np.where(_ok, (_r_t - _f_t) ** 2, np.nan)
            _upd = ~np.isnan(_e2)
            _dmsfe_acc[_upd] = _dl * _dmsfe_acc[_upd] + _e2[_upd]

        # ── Step 3: update weights for period t+1
        _pos = _dmsfe_acc > 0
        if _pos.any():
            _inv = np.where(_pos, 1.0 / np.where(_pos, _dmsfe_acc, 1.0), 0.0)
            _s   = _inv.sum()
            _dmsfe_wt = _inv / _s if _s > 0 else np.ones(_N_dm) / _N_dm

    _col = f"predicted_DMSFE_{int(round(_dl * 100))}"
    _dmsfe_results[_dl] = {"col": _col, "forecast": _preds, "weights": _wts_ts}

    # Merge into eval_df and cer_df
    _tmp = pd.DataFrame({"date": _dm_oos["date"].values, _col: _preds})
    eval_df = pd.merge(eval_df, _tmp, on="date", how="left")
    cer_df  = pd.merge(cer_df,  _tmp, on="date", how="left")

    _n_v = int(np.sum(~np.isnan(_preds)))
    print(f"  delta={_dl:.2f}  col='{_col}'  valid={_n_v}/{_T_dm}")

# ── R²_OS + Clark–West ────────────────────────────────────────────────────
print("\n  R²_OS — DMSFE forecasts")
_dm_r2_rows = []

for _pl, (_ps, _pe) in PERIODS.items():
    _sub = eval_df[(eval_df["date"] >= _ps) & (eval_df["date"] <= _pe)].copy()
    for _dl, _d in _dmsfe_results.items():
        _col = _d["col"]
        if _col not in _sub.columns:
            continue
        _v = _sub[["eqprem", _col, "predicted_HM"]].dropna()
        if len(_v) < 10:
            continue
        _a, _p, _bm    = _v["eqprem"].values, _v[_col].values, _v["predicted_HM"].values
        _r2            = r2_os(_a, _p, _bm)
        _tcw, _pcw     = clark_west(_a, _p, _bm)
        _dm_r2_rows.append({
            "Period":     _pl,
            "Model":      f"DMSFE δ={_dl:.2f}",
            "delta":      _dl,
            "n":          len(_v),
            "R2_OS (%)":  round(_r2 * 100, 2),
            "CW t-stat":  round(_tcw, 3),
            "CW p-value": round(_pcw, 4),
            "Sig":        ("***" if _pcw < 0.01 else "**" if _pcw < 0.05
                           else "*" if _pcw < 0.10 else ""),
        })

_dm_r2_df = pd.DataFrame(_dm_r2_rows)
for _pl in PERIODS:
    _tbl = _dm_r2_df[_dm_r2_df["Period"] == _pl]
    if len(_tbl):
        print(f"\n  {_pl}")
        print(
            _tbl[["Model", "n", "R2_OS (%)", "CW t-stat", "CW p-value"]]
            .to_string(index=False)
        )

# ── CER gain ──────────────────────────────────────────────────────────────
print("\n  CER Gain — DMSFE forecasts")
_dm_cer_rows = []

for _pl, (_ps, _pe) in PERIODS.items():
    _sub = cer_df[(cer_df["date"] >= _ps) & (cer_df["date"] <= _pe)].copy()
    for _dl, _d in _dmsfe_results.items():
        _col = _d["col"]
        if _col not in _sub.columns:
            continue
        _nv = _sub[["simple_eqprem", "sigma2", _col, "predicted_HM"]].dropna().shape[0]
        if _nv < 24:
            continue
        _gain    = _cer_gain_bps(_col, _sub)
        _lo, _hi = _bootstrap_ci(_col, _sub)
        _btc     = _breakeven_tc(_col, _sub)
        _dm_cer_rows.append({
            "Period":                      _pl,
            "Model":                       f"DMSFE δ={_dl:.2f}",
            "delta":                       _dl,
            "n":                           _nv,
            "CER gain (bps)":              round(_gain, 1),
            f"CI lo {int(CI_LEVEL*100)}%": round(_lo, 1),
            f"CI hi {int(CI_LEVEL*100)}%": round(_hi, 1),
            "Breakeven TC (bps)":          round(_btc, 1) if not np.isnan(_btc) else np.nan,
        })

_dm_cer_df = pd.DataFrame(_dm_cer_rows)
for _pl in PERIODS:
    _tbl = _dm_cer_df[_dm_cer_df["Period"] == _pl]
    if len(_tbl):
        print(f"\n  {_pl}")
        print(_tbl.drop(columns="Period").to_string(index=False))

# ── Weight diagnostics ────────────────────────────────────────────────────
print("\n  DMSFE Weight Diagnostics")
_dm_wdiag_rows = []
_dm_dates_idx  = pd.DatetimeIndex(_dm_oos["date"].values)

for _dl, _d in _dmsfe_results.items():
    _wts = _d["weights"]                  # (T, N)
    for _pl, (_ps, _pe) in PERIODS.items():
        _m    = (_dm_dates_idx >= _ps) & (_dm_dates_idx <= _pe)
        _wp   = _wts[_m]                  # (T_p, N)
        # Restrict to rows without NaN (post-burn-in)
        _ok   = ~np.isnan(_wp).any(axis=1)
        if _ok.sum() < 2:
            continue
        _wp = _wp[_ok]
        _dm_wdiag_rows.append({
            "Period":          _pl,
            "delta":           _dl,
            "n":               int(_ok.sum()),
            "avg weight std":  round(float(np.mean(np.std(_wp, axis=1))),   6),
            "avg max weight":  round(float(np.mean(np.max(_wp, axis=1))),   4),
            "avg min weight":  round(float(np.mean(np.min(_wp, axis=1))),   4),
            "avg eff N":       round(float(np.mean(1.0 / np.sum(_wp**2, axis=1))), 2),
        })

_dm_wdiag_df = pd.DataFrame(_dm_wdiag_rows)
_dm_wdiag_sub = _dm_wdiag_df[
    _dm_wdiag_df["Period"].str.contains("Pre-1994|Post-1994|Full \\(1979", regex=True)
].copy()
if not _dm_wdiag_sub.empty:
    print(_dm_wdiag_sub.to_string(index=False))

# ── DMSFE S^fr: rolling Spearman correlation ──────────────────────────────
print("\n  DMSFE S^fr (rolling Spearman, actual returns vs DMSFE forecast)")

_dm_sfr_rows = []

for _dl, _d in _dmsfe_results.items():
    _fc_ser  = pd.Series(
        _d["forecast"], index=pd.DatetimeIndex(_dm_oos["date"].values)
    )
    _act_ser = pd.Series(
        _dm_r, index=pd.DatetimeIndex(_dm_oos["date"].values)
    )
    _T_sfr = len(_act_ser)

    for W in WINDOWS:
        _rows = []
        for _t in range(W, _T_sfr + 1):
            _r_w = _act_ser.iloc[_t - W:_t].values
            _f_w = _fc_ser.iloc[_t - W:_t].values
            _dt  = _act_ser.index[_t - 1]
            if (not np.isnan(_f_w).any()) and np.std(_f_w) > 0:
                _rho, _ = spearmanr(_r_w, _f_w)
            else:
                _rho = np.nan
            _rows.append({"date": _dt, "delta": _dl, "W": W,
                          "S_fr_DMSFE": round(_rho, 6) if not np.isnan(_rho) else np.nan})
        _dm_sfr_rows.extend(_rows)

_dm_sfr_df = pd.DataFrame(_dm_sfr_rows)

# ── Save all DMSFE results to Excel ───────────────────────────────────────
print(f"\n  Saving DMSFE results to {out_path2} ...")

with pd.ExcelWriter(out_path2, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:

    _dm_r2_df.to_excel(writer, sheet_name="DMSFE_R2OS", index=False)
    _dm_cer_df.to_excel(writer, sheet_name="DMSFE_CER", index=False)
    _dm_wdiag_df.to_excel(writer, sheet_name="DMSFE_wdiag", index=False)
    _dm_sfr_df.to_excel(writer, sheet_name="DMSFE_Sfr", index=False)

    # Weight time series — one sheet per delta
    for _dl, _d in _dmsfe_results.items():
        _wts_df = pd.DataFrame(
            _d["weights"], columns=[f"w_{p}" for p in _fc_arr]
        )
        _wts_df.insert(0, "date",     _dm_oos["date"].values)
        _wts_df.insert(1, "forecast", _d["forecast"])
        _sheet = f"DMSFE_w_{int(round(_dl*100))}"
        _wts_df.to_excel(writer, sheet_name=_sheet[:31], index=False)

print("  DMSFE results saved.")
print("\nDMSFE combination weights complete.")

##################################################
# Discounted OLS (DOLS) Bivariate + 1/N         #
# Exponential decay WLS  delta in {0.97,0.99,0.995} #
##################################################
#
# For each predictor i and OOS origin t, estimate bivariate WLS:
#   r_{s+1} = alpha_i + beta_i * x_{i,s} + eps
# with weights w_s = delta^(t-s) / sum_j delta^(t-j)  (normalised).
# Most recent observation gets weight proportional to 1.
#
# 1/N combination: r_hat_DOLS_{t+1} = (1/N) * sum_i r_hat_{i,t+1}
#
# CT restriction: wrong-sign beta set to 0 (restricted).
# Non-negative constraint: max(0, y_hat) at individual level.

print("\n" + "=" * 60)
print("Discounted OLS (DOLS)  [exponential decay WLS, restricted]")
print("=" * 60)

DOLS_DELTAS = [0.98851, 0.99424, 0.99616]

# ── OOS window ────────────────────────────────────────────────────────────
_dols_oos = (
    eval_df[
        (eval_df["date"] >= FORECAST_START) & (eval_df["date"] <= FULL_END)
    ]
    .copy()
    .reset_index(drop=True)
)
_T_dols         = len(_dols_oos)
_N_dols         = len(forecast_cols)
_oos_dates_dols = _dols_oos["date"].values   # (T_dols,)

print(f"\n  OOS window: T={_T_dols} "
      f"({_dols_oos['date'].iloc[0].strftime('%Y-%m')} – "
      f"{_dols_oos['date'].iloc[-1].strftime('%Y-%m')}), "
      f"N={_N_dols} predictors, deltas={DOLS_DELTAS}")

# Pre-build sorted numpy arrays for each predictor
_dols_dates_arr = {}
_dols_y_arr     = {}
_dols_x_arr     = {}

for _p in forecast_cols:
    _dd = df[["date", "eqprem", _p]].dropna().copy()
    _lc = f"{_p}_lag"
    _dd[_lc] = _dd[_p].shift(1)
    _dd = _dd.dropna(subset=[_lc]).sort_values("date").reset_index(drop=True)
    _dols_dates_arr[_p] = _dd["date"].values
    _dols_y_arr[_p]     = _dd["eqprem"].values
    _dols_x_arr[_p]     = _dd[_lc].values

# Forecast matrices: _dols_P_all[delta][t, i]
_dols_P_all = {_dl: np.full((_T_dols, _N_dols), np.nan) for _dl in DOLS_DELTAS}

_dols_start = time.time()

for _i, _p in enumerate(forecast_cols):
    _di = _dols_dates_arr[_p]
    _yi = _dols_y_arr[_p]
    _xi = _dols_x_arr[_p]
    _exp_sign_p = EXPECTED_SIGNS.get(_p, 0)   # expected beta sign for CT restriction

    for _t in range(_T_dols):
        _oos_dt = _oos_dates_dols[_t]

        # Training: all observations strictly before _oos_dt
        _end = int(np.searchsorted(_di, _oos_dt, side="left"))
        if _end < 5:
            continue

        _y_tr = _yi[:_end]
        _x_tr = _xi[:_end]

        # Forecast predictor value (lagged) at _oos_dt
        _pt = int(np.searchsorted(_di, _oos_dt, side="left"))
        if _pt >= len(_di) or _di[_pt] != _oos_dt:
            continue
        _x_pred = float(_xi[_pt])

        # WLS for each delta (training data fixed; only weights differ)
        _exps_base = np.arange(_end - 1, -1, -1, dtype=np.float64)
        for _dl in DOLS_DELTAS:
            _w  = _dl ** _exps_base
            _w /= _w.sum()          # normalise for numerical stability
            _sw = np.sqrt(_w)

            _Xw = np.column_stack([_sw, _x_tr * _sw])
            _yw = _y_tr * _sw
            try:
                _coef, _, _, _ = np.linalg.lstsq(_Xw, _yw, rcond=None)
            except Exception:
                continue

            _alpha_dl = float(_coef[0])
            _beta_dl  = float(_coef[1])
            # CT restriction: wrong-sign beta → set to 0
            if ((_beta_dl > 0 and _exp_sign_p < 0) or
                    (_beta_dl < 0 and _exp_sign_p > 0)):
                _beta_dl = 0.0
            _yh = max(0.0, _alpha_dl + _beta_dl * _x_pred)
            _dols_P_all[_dl][_t, _i] = _yh

print(f"  WLS loops complete ({time.time() - _dols_start:.1f}s)")

# ── 1/N combination + merge into eval_df / cer_df ─────────────────────────
_dols_results = {}

for _dl in DOLS_DELTAS:
    _fc_1n = np.nanmean(_dols_P_all[_dl], axis=1)
    _col   = f"predicted_DOLS_{int(round(_dl * 1000))}"
    _dols_results[_dl] = {"col": _col, "forecast_1n": _fc_1n,
                           "matrix": _dols_P_all[_dl]}

    _tmp = pd.DataFrame({"date": _oos_dates_dols, _col: _fc_1n})
    eval_df = pd.merge(eval_df, _tmp, on="date", how="left")
    cer_df  = pd.merge(cer_df,  _tmp, on="date", how="left")

    _nv = int(np.sum(~np.isnan(_fc_1n)))
    print(f"  delta={_dl}: {_nv}/{_T_dols} valid 1/N forecasts  col={_col}")

# ── R²_OS + Clark–West ────────────────────────────────────────────────────
print("\n  R²_OS — DOLS forecasts")
_dols_r2_rows = []

for _pl, (_ps, _pe) in PERIODS.items():
    _sub = eval_df[(eval_df["date"] >= _ps) & (eval_df["date"] <= _pe)].copy()
    for _dl, _dd in _dols_results.items():
        _col = _dd["col"]
        if _col not in _sub.columns:
            continue
        _v = _sub[["eqprem", _col, "predicted_HM"]].dropna()
        if len(_v) < 10:
            continue
        _a, _p2, _bm = _v["eqprem"].values, _v[_col].values, _v["predicted_HM"].values
        _r2          = r2_os(_a, _p2, _bm)
        _tcw, _pcw   = clark_west(_a, _p2, _bm)
        _sig = ("***" if _pcw < 0.01 else "**" if _pcw < 0.05
                else "*"   if _pcw < 0.10 else "")
        _dols_r2_rows.append({
            "Period":     _pl,
            "Model":      f"DOLS \u03b4={_dl:.3f}",
            "delta":      _dl,
            "n":          len(_v),
            "R2_OS (%)":  round(_r2 * 100, 2),
            "CW t-stat":  round(_tcw, 3),
            "CW p-value": round(_pcw, 4),
            "Sig":        _sig,
        })

_dols_r2_df = pd.DataFrame(_dols_r2_rows)
for _pl in PERIODS:
    _tbl = _dols_r2_df[_dols_r2_df["Period"] == _pl]
    if len(_tbl):
        print(f"\n  {_pl}")
        print(_tbl.drop(columns="Period").to_string(index=False))

# ── CER Gain ──────────────────────────────────────────────────────────────
print("\n  CER Gain — DOLS forecasts")
_dols_cer_rows = []

for _pl, (_ps, _pe) in PERIODS.items():
    _sub = cer_df[(cer_df["date"] >= _ps) & (cer_df["date"] <= _pe)].copy()
    for _dl, _dd in _dols_results.items():
        _col = _dd["col"]
        if _col not in _sub.columns:
            continue
        _nv = _sub[["simple_eqprem", "sigma2", _col, "predicted_HM"]].dropna().shape[0]
        if _nv < 24:
            continue
        _gain    = _cer_gain_bps(_col, _sub)
        _lo, _hi = _bootstrap_ci(_col, _sub)
        _btc     = _breakeven_tc(_col, _sub)
        _dols_cer_rows.append({
            "Period":                      _pl,
            "Model":                       f"DOLS \u03b4={_dl:.3f}",
            "delta":                       _dl,
            "n":                           _nv,
            "CER gain (bps)":              round(_gain, 1),
            f"CI lo {int(CI_LEVEL*100)}%": round(_lo, 1),
            f"CI hi {int(CI_LEVEL*100)}%": round(_hi, 1),
            "Breakeven TC (bps)":          round(_btc, 1) if not np.isnan(_btc) else np.nan,
        })

_dols_cer_df = pd.DataFrame(_dols_cer_rows)
for _pl in PERIODS:
    _tbl = _dols_cer_df[_dols_cer_df["Period"] == _pl]
    if len(_tbl):
        print(f"\n  {_pl}")
        print(_tbl.drop(columns="Period").to_string(index=False))

# ── DOLS S^fr (rolling Spearman) ──────────────────────────────────────────
print("\n  DOLS S^fr (rolling Spearman, actual returns vs DOLS 1/N forecast)")

_dols_sfr_rows = []

for _dl, _dd in _dols_results.items():
    _fc_dols = _dd["forecast_1n"]
    for W in WINDOWS:
        for _t_idx in range(_T_dols):
            if _t_idx < W - 1:
                continue
            _dt    = pd.Timestamp(_oos_dates_dols[_t_idx])
            _r_win = _dols_oos["eqprem"].values[_t_idx - W + 1: _t_idx + 1]
            _f_win = _fc_dols[_t_idx - W + 1: _t_idx + 1]
            _ok    = ~(np.isnan(_r_win) | np.isnan(_f_win))
            if _ok.sum() >= 5 and np.std(_f_win[_ok]) > 0:
                _rho, _ = spearmanr(_r_win[_ok], _f_win[_ok])
            else:
                _rho = np.nan
            _dols_sfr_rows.append({
                "date":      _dt,
                "delta":     _dl,
                "W":         W,
                "S_fr_DOLS": round(_rho, 6) if not np.isnan(_rho) else np.nan,
            })

_dols_sfr_df = pd.DataFrame(_dols_sfr_rows)

# ── Save to Excel ─────────────────────────────────────────────────────────
print(f"\n  Saving DOLS results to {out_path2} ...")

with pd.ExcelWriter(out_path2, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    _dols_r2_df.to_excel(writer,  sheet_name="DOLS_R2OS", index=False)
    _dols_cer_df.to_excel(writer, sheet_name="DOLS_CER",  index=False)
    _dols_sfr_df.to_excel(writer, sheet_name="DOLS_Sfr",  index=False)

    for _dl, _dd in _dols_results.items():
        _mat_df = pd.DataFrame(
            _dd["matrix"], columns=[f"pred_{_p}" for _p in forecast_cols]
        )
        _mat_df.insert(0, "date",        _oos_dates_dols)
        _mat_df.insert(1, "forecast_1n", _dd["forecast_1n"])
        _sheet = f"DOLS_{int(round(_dl * 1000))}"
        _mat_df.to_excel(writer, sheet_name=_sheet[:31], index=False)

print("  DOLS results saved.")

# ── DOLS S^p (rolling pairwise Spearman among individual DOLS forecasts) ──
print("\n  DOLS S^p (rolling pairwise Spearman among individual forecasts)")

_dols_sp_rows = []

for _dl in DOLS_DELTAS:
    _P_dl = _dols_P_all[_dl]   # (T, N)
    for W in WINDOWS:
        for _t_idx in range(_T_dols):
            if _t_idx < W - 1:
                continue
            _dt      = pd.Timestamp(_oos_dates_dols[_t_idx])
            _P_win   = _P_dl[_t_idx - W + 1: _t_idx + 1, :]   # (W, N)
            _ok_cols = ~np.isnan(_P_win).any(axis=0)
            _n_ok    = _ok_cols.sum()
            if _n_ok < 2:
                _sp_val = np.nan
            else:
                _P_ok    = _P_win[:, _ok_cols]
                _P_ranked = np.apply_along_axis(rankdata, 0, _P_ok)
                _corr_mat = np.corrcoef(_P_ranked.T)
                _upper    = np.triu_indices(_n_ok, k=1)
                _sp_val   = float(np.nanmean(_corr_mat[_upper]))
            _dols_sp_rows.append({
                "date":      _dt,
                "delta":     _dl,
                "W":         W,
                "S_p_DOLS":  round(_sp_val, 6) if not np.isnan(_sp_val) else np.nan,
            })

_dols_sp_df = pd.DataFrame(_dols_sp_rows)

with pd.ExcelWriter(out_path2, engine="openpyxl", mode="a",
                    if_sheet_exists="replace") as writer:
    _dols_sp_df.to_excel(writer, sheet_name="DOLS_Sp", index=False)

print("  DOLS S^p saved.")

# ── DOLS S^fr plots (one per delta) ───────────────────────────────────────
print("\n  Plotting DOLS S^fr and S^p ...")

_CB_COLORS_DOLS = ["#0072B2", "#D55E00", "#009E73"]
_CB_STYLES_DOLS = ["-", "--", ":"]

for _dl in DOLS_DELTAS:
    _sub_sfr = _dols_sfr_df[_dols_sfr_df["delta"] == _dl]

    fig, ax = plt.subplots(figsize=(11, 4))
    for idx, W in enumerate(WINDOWS):
        _w_sub = _sub_sfr[_sub_sfr["W"] == W].sort_values("date")
        ax.plot(_w_sub["date"], _w_sub["S_fr_DOLS"],
                color=_CB_COLORS_DOLS[idx], linestyle=_CB_STYLES_DOLS[idx],
                linewidth=1.3, label=f"W={W} months")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("S$^{fr}$ (DOLS)", fontsize=14)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(left=pd.Timestamp("1979-01-01"), right=pd.Timestamp("2025-01-01"))
    plt.xticks(rotation=45)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _fname = f"plot_DOLS_S_fr_delta{int(round(_dl*100000))}_monthly_restricted.png"
    fig.savefig(_fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {_fname}")

for _dl in DOLS_DELTAS:
    _sub_sp = _dols_sp_df[_dols_sp_df["delta"] == _dl]

    fig, ax = plt.subplots(figsize=(11, 4))
    for idx, W in enumerate(WINDOWS):
        _w_sub = _sub_sp[_sub_sp["W"] == W].sort_values("date")
        ax.plot(_w_sub["date"], _w_sub["S_p_DOLS"],
                color=_CB_COLORS_DOLS[idx], linestyle=_CB_STYLES_DOLS[idx],
                linewidth=1.3, label=f"W={W} months")
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("S$^{p}$ (DOLS)", fontsize=14)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(left=pd.Timestamp("1979-01-01"), right=pd.Timestamp("2025-01-01"))
    plt.xticks(rotation=45)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _fname = f"plot_DOLS_S_p_delta{int(round(_dl*100000))}_monthly_restricted.png"
    fig.savefig(_fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {_fname}")

print("  DOLS S^fr / S^p plots saved.")
print("\nDiscounted OLS (DOLS) complete.")

print("  Saved: plot_pred_vs_eqprem_1994_2000_monthly_restricted.png")

##################################################
# Return Variance Over Time (all windows)        #
##################################################

print("\nPlotting return variance over time ...")

_RV_COLORS = ["#0072B2", "#D55E00", "#009E73"]
_RV_STYLES = ["-", "--", ":"]

fig, ax = plt.subplots(figsize=(11, 4))
for idx, W in enumerate(WINDOWS):
    _rv = pd.Series(actual_vec).rolling(W).var().values
    _mask = ~np.isnan(_rv)
    ax.plot(np.array(dates_oos)[_mask], _rv[_mask],
            color=_RV_COLORS[idx], linestyle=_RV_STYLES[idx],
            linewidth=1.3, label=f"W={W} months")

ax.set_xlabel("Date", fontsize=14)
ax.set_ylabel("Variance", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.set_xlim(left=pd.Timestamp("1979-01-01"), right=pd.Timestamp("2025-01-01"))
plt.xticks(rotation=45)
ax.legend(fontsize=13)
ax.grid(True, alpha=0.3)
fig.tight_layout()
_fname_png = "plot_return_variance_monthly_restricted.png"
_fname_pdf = "plot_return_variance_monthly_restricted.pdf"
fig.savefig(_fname_png, dpi=300, bbox_inches="tight")
fig.savefig(_fname_pdf, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {_fname_png} + {_fname_pdf}")

##################################################
# Individual Predictor–Return Correlation Plots  #
# (Spearman, colorblind-friendly)                #
##################################################

print("\nComputing individual Spearman predictor–return correlations ...")

# Wong (2011) colorblind-safe palette (8 colours)
_WONG = [
    "#000000",   # black
    "#E69F00",   # orange
    "#56B4E9",   # sky blue
    "#009E73",   # bluish green
    "#F0E442",   # yellow
    "#0072B2",   # blue
    "#D55E00",   # vermillion
    "#CC79A7",   # reddish purple
]
# 8 colours × 2 linestyles = 16 unique combos (covers 14 predictors)
_IPC_COLORS = (_WONG * 2)[:len(forecast_cols)]
_IPC_STYLES = (["-"] * len(_WONG) + ["--"] * len(_WONG))[:len(forecast_cols)]

for W in WINDOWS:
    print(f"  Window = {W} months ...")
    _ipc_rows = []

    for t in range(W, T + 1):
        r_window = actual_vec[t - W:t]
        P_window = pred_matrix[t - W:t, :]
        date     = dates_oos[t - 1]

        row = {"date": date}
        for i, pred in enumerate(forecast_cols):
            p_i = P_window[:, i]
            if np.std(p_i) > 0 and np.std(r_window) > 0:
                row[pred] = float(spearmanr(r_window, p_i).correlation)
            else:
                row[pred] = np.nan
        _ipc_rows.append(row)

    _ipc_df = pd.DataFrame(_ipc_rows)

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(13, 6))

    for i, pred in enumerate(forecast_cols):
        ax.plot(_ipc_df["date"], _ipc_df[pred],
                color=_IPC_COLORS[i], linestyle=_IPC_STYLES[i],
                linewidth=1.0, label=pred)

    ax.axhline(0, color="lightgray", linewidth=1.0, zorder=0)
    ax.axvline(pd.Timestamp("1994-01-01"), color="gray", linewidth=0.9,
               linestyle="--", alpha=0.7, label="1994 break")

    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("Correlation", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(left=pd.Timestamp("1979-01-01"), right=pd.Timestamp("2025-01-01"))
    plt.xticks(rotation=45)
    ax.grid(True, alpha=0.3)

    ax.legend(fontsize=9, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), frameon=True, fancybox=False)

    fig.tight_layout(rect=[0, 0.12, 1, 1])
    _fname_png = f"individual_predictor_corr_monthly_restricted_W{W}.png"
    _fname_pdf = f"individual_predictor_corr_monthly_restricted_W{W}.pdf"
    fig.savefig(_fname_png, dpi=300, bbox_inches="tight")
    fig.savefig(_fname_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {_fname_png} + {_fname_pdf}")

print("Individual predictor correlation plots complete.")