# fraud-detection

**Nova Credit: Fraudulent Transaction Detection for Digital Money Transfer**

This project aims to make Nova's transaction security stronger by detecting fraudulent transactions at the point of processing. It develops a machine-learning model that provides fast, data-driven fraud predictions to support accurate detection, regulatory compliance, and improved customer experience.

## Project structure

```
fraud-detection/
├── data/
│   └── nova_pay_combined.csv   # raw transaction dataset
├── notebook/
│   └── 01_EDA.ipynb            # exploratory data analysis & initial cleaning
└── README.md
```

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

## Next steps
The following issues have been identified but not yet fixed:
- [ ] Remove the 200 duplicate rows
- [ ] Strip whitespace, standardise case, and fix typos in `channel`, `kyc_tier`, `home_country` and `ip_country`, and merge the `NAN` / `nan` strings into `Unknown`
- [ ] Strip commas from `amount_src` and convert it to a number
- [ ] Parse `timestamp` to datetime and handle the invalid and missing timestamps
- [ ] Handle the sentinel and out-of-range values (`fee`, `device_trust_score`, `ip_risk_score`, `txn_velocity_1h`)
- [ ] Consider adding a missing-value indicator feature, because missingness appears together across columns
- [ ] Feature engineering (time-of-day, corridor, customer/device aggregates)
- [ ] Model training with class-imbalance handling, and evaluation (precision/recall, PR-AUC)
