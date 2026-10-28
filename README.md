# fraud-detection

**Nova Credit: Fraudulent Transaction Detection for Digital Money Transfer**

This project aims to make Nova's transaction security stronger by detecting fraudulent transactions at the point of processing. It develops a machine-learning model that provides fast, data-driven fraud predictions to support accurate detection, regulatory compliance, and improved customer experience.

## Project structure

```
fraud-detection/
├── data/
│   └── nova_pay_combined.csv   # raw transaction dataset
├── models/
│   └── fraud_model.joblib      # saved champion model (created by the notebook)
├── notebook/
│   └── 01_EDA.ipynb            # EDA, cleaning, feature engineering, models, SHAP
├── src/
│   └── app.py                  # FastAPI app that serves the champion model
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv fraud-venv
fraud-venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

The pinned versions in `requirements.txt` support Python 3.7, which is the version `fraud-venv` uses.

## Dataset

`data/nova_pay_combined.csv` has **11,400 rows and 26 columns**. Each row is one complete transaction.

| Group | Columns |
|---|---|
| Identifiers | `transaction_id`, `customer_id`, `device_id`, `ip_address` |
| Transaction | `timestamp`, `channel`, `source_currency`, `dest_currency`, `amount_src`, `amount_usd`, `fee`, `exchange_rate_src_to_dest` |
| Customer / account | `home_country`, `kyc_tier`, `account_age_days`, `chargeback_history_count` |
| Device / network | `new_device`, `device_trust_score`, `ip_country`, `location_mismatch`, `ip_risk_score` |
| Risk & behaviour | `risk_score_internal`, `txn_velocity_1h`, `txn_velocity_24h`, `corridor_risk` |
| Target | `is_fraud` (1 = fraud, 0 = legitimate) |

- **Class imbalance:** about **8.7%** of transactions are fraudulent (mean of `is_fraud` = 0.0875).
- **Date range:** October 2022 to November 2025.
- **Corridors:** source currencies are USD, GBP and CAD. Destination currencies are NGN, USD, INR, CNY, PHP, GBP, CAD, EUR and MXN.

## Exploratory Data Analysis (`notebook/01_EDA.ipynb`)

### 1. Structure and summary statistics
- Inspected the first and last rows, data types (`info()`), and `describe()` statistics.
- `amount_src` is stored as text (`object`) instead of a number because some values contain thousands separators.

### 2. Missing values

| Column | Missing (%) |
|---|---|
| `amount_usd` | 2.68 |
| `ip_address` | 2.68 |
| `ip_country` | 2.64 |
| `kyc_tier` | 2.63 |
| `fee` | 2.59 |
| `device_trust_score` | 2.59 |
| `timestamp` | 0.25 |

**Missingness pattern:** the gaps between missing `ip_address` rows are irregular, so the values are not missing on a fixed schedule. However, missing values **appear together**: of the 305 rows missing `amount_usd`, 295 are also missing `fee`, 300 are missing `kyc_tier`, and 295 are missing `device_trust_score`. This points to a system or ingestion fault that drops several fields at once, not to random gaps.

### 3. Duplicates
- Found **200 fully duplicated rows**. The same pattern shows up as repeated `transaction_id`s: there are 11,200 unique IDs across 11,400 rows.

### 4. Outliers and invalid values
I drew a boxplot for each numeric column. Several values are outside the valid range or look like placeholder (sentinel) values.


### 5. Categorical data quality audit
A value-count audit of every text column, plus a spell-check pass with `pyspellchecker`, found inconsistent labels.


### 6. Non-numeric values in numeric columns
- `amount_src` contains comma-formatted values such as `9,998.85`, `9,995.95`, `1,541.55` and `9,991.24`. These do not parse as numbers.

### 7. Missing-value treatment (done so far)
- **Median imputation** for `amount_usd`, `fee` and `device_trust_score`.
- **Filled with `'Unknown'`:** `ip_address`, `ip_country` and `kyc_tier`.

## Fraud Detection API (`src/app.py`)

A FastAPI app serves the champion model (tuned XGBoost). It cleans each incoming transaction the same way as the notebook, scores it, and returns the top 3 SHAP reasons.

**1. Create the model file.** Run all cells in `notebook/01_EDA.ipynb`. The "Save the Champion Model" cell writes `models/fraud_model.joblib`.

**2. Start the API** from the project root:

```bash
fraud-venv\Scripts\activate
uvicorn src.app:app --reload
```

**3. Try it.** Open http://127.0.0.1:8000/docs for an interactive page where you can send test requests.

| Endpoint | Method | What it does |
|---|---|---|
| `/health` | GET | Checks that the API is running and shows the model name and threshold |
| `/predict` | POST | Scores one transaction |
| `/predict/batch` | POST | Scores a list of up to 1,000 transactions |

Example request to `/predict`:

```json
{
  "transaction_id": "54f00f78-94aa-4a08-bf6b-12877aa47186",
  "timestamp": "2025-11-26T07:09:56+00:00",
  "home_country": "CA",
  "source_currency": "CAD",
  "dest_currency": "NGN",
  "channel": "web",
  "amount_src": 537.17,
  "amount_usd": 397.51,
  "fee": 8.86,
  "exchange_rate_src_to_dest": 822.22,
  "new_device": false,
  "ip_country": "UK",
  "location_mismatch": true,
  "ip_risk_score": 0.891,
  "kyc_tier": "low",
  "account_age_days": 9,
  "device_trust_score": 0.173,
  "chargeback_history_count": 2,
  "risk_score_internal": 0.73,
  "txn_velocity_1h": 3,
  "txn_velocity_24h": 6,
  "corridor_risk": 0.0
}
```

The response has the fraud probability, the decision (`is_fraud` is true when the probability is 0.5 or higher), and the top 3 reasons:

```json
{
  "transaction_id": "54f00f78-94aa-4a08-bf6b-12877aa47186",
  "fraud_probability": 0.9744,
  "is_fraud": true,
  "top_reasons": [
    {"feature": "txn_velocity_1h", "impact": 1.5333},
    {"feature": "txn_velocity_24h", "impact": 1.0548},
    {"feature": "ip_risk_score", "impact": 0.3171}
  ]
}
```

`amount_usd`, `fee`, `device_trust_score`, `ip_country` and `kyc_tier` can be left out or sent as `null`. Missing numbers are filled with the training medians, and missing text becomes `unknown`.

## Next steps
- [x] Remove the 200 duplicate rows
- [x] Strip whitespace, standardise case, and fix typos in `channel`, `kyc_tier`, `home_country` and `ip_country`, and merge the `NAN` / `nan` strings into `unknown`
- [x] Strip commas from `amount_src` and convert it to a number
- [x] Parse `timestamp` to datetime and handle the invalid and missing timestamps
- [x] Handle the sentinel and out-of-range values (`fee`, `device_trust_score`, `ip_risk_score`, `txn_velocity_1h`)
- [x] Feature engineering (time features, `fee_percent`, `is_new_account`)
- [x] Baseline logistic regression, and tuned XGBoost and LightGBM, with class-imbalance handling
- [x] SHAP explanations for the champion model
- [x] FastAPI app to serve the champion model
- [ ] Confirm `risk_score_internal` is not calculated from the fraud outcome (possible label leakage)
- [ ] Consider adding a missing-value indicator feature, because missingness appears together across columns
- [ ] Tune the decision threshold based on the cost of missed fraud vs false alarms
