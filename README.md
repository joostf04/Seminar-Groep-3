# Equity Premium Predictability — OLS Expanding Window

Out-of-sample equity premium forecasting using OLS expanding-window regressions across monthly and quarterly data, with and without Campbell & Thompson (2008) sign restrictions.

---

## Structure

```text
├── Data_Seminar.xlsx                              ← dataset
├── Rolling Window RESTRICTED.py
├── Rolling Window UNRESTRICTED.py
├── ols_expanding_window_monthly_unrestricted.py
├── ols_expanding_window_monthly_restricted.py
├── ols_expanding_window_quarterly_unrestricted.py
├── ols_expanding_window_quarterly_restricted.py
├── PredEQpremPlots.py
└── README.md
```

---

## Files

| File | Description |
|---|---|
| `ols_expanding_window_quarterly_unrestricted.py` | Quarterly forecasts, no restrictions |
| `ols_expanding_window_quarterly_restricted.py` | Quarterly forecasts, CT sign restrictions |
| `ols_expanding_window_monthly_unrestricted.py` | Monthly forecasts, no restrictions |
| `ols_expanding_window_monthly_restricted.py` | Monthly forecasts, CT sign restrictions |
| `Rolling Window RESTRICTED.py` | Rolling window forecasts with CT sign restrictions |
| `Rolling Window UNRESTRICTED.py` | Rolling window forecasts without restrictions |
| `PredEQpremPlots.py` | Script for generating positive and negative predictor average together with equity premium |


The rolling window scripts contain the rolling window estimation results, complementing the expanding window approach.
---

## Requirements

```
pip install pandas numpy statsmodels scipy scikit-learn matplotlib openpyxl
```

---

## How to Run

Place `Data_Seminar.xlsx` in the project folder, then run any script:

```
python "ols_expanding_window_monthly_unrestricted.py"
python "ols_expanding_window_monthly_restricted.py"
python "ols_expanding_window_quarterly_unrestricted.py"
python "ols_expanding_window_quarterly_restricted.py"
```

