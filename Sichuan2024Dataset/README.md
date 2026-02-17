
# Sichuan2024Dataset (Anonymized Real-World Subset)

**Sichuan2024Dataset** is an open-access, anonymized subset of real-world smart meter data, collected from three representative districts (referred to as City-A, City-B, City-C) in Sichuan Province, China, covering the full year of 2024.

This dataset was specifically curated to evaluate dynamic clustering methods (e.g., DynEC) using actual consumption patterns influenced by the 2024 Sichuan Time-of-Use (TOU) pricing policy. Unlike purely unlabeled raw data, this subset includes **verified ground-truth labels** derived from utility account metadata and on-site validation, enabling the direct and rigorous computation of clustering metrics such as ARI (Adjusted Rand Index).

## Contents

```text
Sichuan2024Dataset/
  ├── meta.json
  ├── City-A/
  │   ├── users.csv
  │   ├── events.csv
  │   └── profiles_daily.csv
  ├── City-B/
  │   ├── users.csv
  │   ├── events.csv
  │   └── profiles_daily.csv
  └── City-C/
      ├── users.csv
      ├── events.csv
      └── profiles_daily.csv
```

## Key Properties

- **Source:** Real smart meter records from Sichuan Province grid database.
- **Time Span:** `2024-01-01` to `2024-12-31` (366 days).
- **Temporal Resolution:** Daily profiles with 24 hourly readings (`h0`..`h23`).
- **Scale:** Curated subsets of users selected for high data quality and representativeness:
  - **City-A:** 800 users (Mixed residential/commercial)
  - **City-B:** 500 users (Predominantly residential)
  - **City-C:** 650 users (Industrial park zone)
- **Privacy & Preprocessing:** User IDs are hashed. Load values are min-max normalized to `[0, 1]` per row to preserve **shape patterns** while masking sensitive absolute consumption levels.
- **Context:** Reflects real user demand response to the Sichuan TOU policy (Version "2023", effective throughout 2024).

## File Formats

### 1. `profiles_daily.csv` (Per City)

**Description:** The core load profile data containing time-series information.
**Row Granularity:** User-Day

| Column | Type | Description |
| :--- | :--- | :--- |
| `city` | String | Anonymized region code (`City-A`, `City-B`, `City-C`). |
| `user_id` | Integer | Anonymized unique user identifier. |
| `day_index` | Integer | Day index (0 to 365, where 0 is Jan 1st). |
| `date` | String | Calendar date `YYYY-MM-DD`. |
| `true_cluster` | Integer | **Validated User Category**. The ground-truth cluster ID (0-7). This label is derived from utility account metadata (e.g., specific industry codes, residential sub-categories) and serves as the stable ground truth for evaluation. |
| `h0`..`h23` | Float | 24 hourly load values, normalized to `[0, 1]`. |

### 2. `users.csv` (Per City)

**Description:** Static and semi-static attributes for each user, extracted from utility metadata.
**Row Granularity:** User

| Column | Type | Description |
| :--- | :--- | :--- |
| `user_id` | Integer | Unique user identifier. |
| `user_type` | String | Broad category registered in the system (e.g., `residential`, `commercial`, `ev_station`). |
| `base_cluster` | Integer | The user’s dominant behavior mode identified at the start of the year. |
| `amplitude` | Float | **Average Load Magnitude**. A scalar representing the user's typical absolute consumption level (retained before normalization to allow magnitude-aware analysis if needed). |
| `temp_group` | Integer | The local micro-climate zone ID the user belongs to (linked to weather station data). |
| `event_type` | String | Documented infrastructure change or behavior shift (e.g., `EV` charger installation, `PV` integration). Empty if stable. |
| `event_start_day` | Integer | The recorded date index of the infrastructure change (`-1` if no event). |
| `tou_response_strength` | Float | Calculated elasticity score (0–1) indicating the user's responsiveness to peak/valley pricing. |

### 3. `events.csv` (Per City)

**Description:** A registry of significant, verified changes in user consumption mode.
**Row Granularity:** Event

This file lists users where a structural change in electricity consumption was **verified** (e.g., via service change requests, meter upgrades, or documented appliance installation). This provides the "ground truth" for dynamic evolution.

| Column | Type | Description |
| :--- | :--- | :--- |
| `user_id` | Integer | Unique user identifier. |
| `event_type` | String | The category of the verified change (e.g., `EV`, `PV`, `SHOCK`). |
| `start_day` | Integer | Day index when the change became effective. |

### 4. `meta.json`

Contains dataset-level metadata for reproducibility:
- `source_description`: Details on data collection and anonymization protocols.
- `sampling_strategy`: Explanation of how this subset was selected (e.g., stratified sampling based on user types).
- `tou_policy_reference`: Link to the official Sichuan Development and Reform Commission policy document.
- `weather_station_id`: IDs of the weather stations used to correlate temperature data.

## Sichuan TOU Policy Context

The data patterns in this dataset are physically driven by the specific pricing periods in Sichuan Province. Algorithms should be able to detect features aligning with these windows:

- **Residential Valley (Low Price):** `23:00 – 07:00` (Incentivizes night-time usage, e.g., EV charging).
- **Critical Peak Summer (High Price):** Jul–Aug `15:00 – 17:00` (driven by air conditioning load).
- **Critical Peak Winter (High Price):** Dec–Jan `19:00 – 21:00` (driven by heating load).

## Recommended Usage

This dataset is intended to benchmark dynamic clustering algorithms on **real-world noisy data** where ground truth is known via external verification.

### Python Example

```python
import pandas as pd

# 1. Load the real-world subset for City A
df = pd.read_csv("City-A/profiles_daily.csv")
hour_cols = [f"h{i}" for i in range(24)]

# 2. Process dates
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.to_period("M").astype(str)

# 3. Aggregate to monthly patterns for stability analysis
# Group by Month and User to get average profiles
X_month = df.groupby(["month", "user_id"])[hour_cols].mean()

# 4. Extract Ground Truth
# We use the verified metadata label ('true_cluster') for ARI calculation
y_month = df.groupby(["month", "user_id"])["true_cluster"].first()

print(f"Data loaded: {len(df)} daily profiles.")
print(f"Unique Users: {df['user_id'].nunique()}")
```
