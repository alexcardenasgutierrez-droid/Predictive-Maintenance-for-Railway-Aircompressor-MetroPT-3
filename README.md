# 🚆 Predictive Maintenance for Railway Air Compressor — MetroPT-3

> Predicting air compressor failures in metro trains using real sensor data, machine learning, and imbalanced classification techniques.

---

## 📋 Short Description

This project applies machine learning to predict failures in the **Air Production Unit (APU)** of a metro train compressor. The APU is a safety-critical system — it powers the brakes, doors, and suspension. An undetected air leak can force an emergency stop, putting passengers at risk. Using real operational sensor data from Porto's metro system, we built a Random Forest classifier capable of detecting failure events with a **Recall of 69.1%** and an **F1-score of 63.4%**.

---

## 📦 Dataset Description

- **Source:** [MetroPT-3 Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/791/metropt-3+dataset)
- **Origin:** Real metro train in Porto, Portugal (2020)
- **Size:** 15,169,948 rows × 17 columns
- **Period:** February – August 2020, logged at 1Hz (~every second)
- **Sensors:**
  - 8 digital sensors → pressure (bar), temperature (°C), motor current (A)
  - 7 analog sensors → binary signals (0/1) for valves, towers, oil level
  - 1 timestamp column
  - 1 target column: `failure` (binary) that is not given but self-built from the failures records
- **Failures:** 4 real air leak events, all labeled as **High Stress**
- **Key quirk:** Severe class imbalance — only **1.97%** of rows are failures

---

## 🎯 Research Goal

> **How do we handle extreme class imbalance (1.97% failures)?**

Secondary questions:
- Which sensors are most predictive of a failure?
- Can we detect air compressor failures in a metro train before or as they happen, using sensor readings alone?
- What threshold should we use for binary classification in this safety-critical context?

---

## 🔬 Steps Taken

### 1. Exploratory Data Analysis (EDA)
- Visualized all sensor readings over time with failure periods highlighted
- Analyzed sensor behavior in the 3 days before each failure event
- Compared distributions across three periods: **normal**, **pre-failure**, and **failure**
- Built a correlation matrix to identify relationships between sensors and the target

### 2. Data Preprocessing
- Chronological train/test split (70% train, 30% test) — **no shuffling** to avoid data leakage
- Memory optimization: `float64 → float32`, integer binary columns → `bool`
- Resampled full dataset to 1-minute intervals for faster experimentation

### 3. Handling Class Imbalance
Tested multiple strategies:
- **Baseline** — no resampling, threshold tuning only
- **Undersampling 50/50 and 70/30** — removing majority class samples
- **SMOTE** — generating synthetic minority samples ⭐ best result
- **SMOTETomek** — SMOTE + cleaning noisy border samples

### 4. Feature Engineering
- Dropped low-importance features: `second`, `minute`, `Pressure_switch`, `Oil_level`, `Caudal_impulses`
- Dropped highly correlated features: `TP3`, `LPS`, `MPG`, `COMP`
- Final feature set: `TP2`, `H1`, `DV_pressure`, `Reservoirs`, `Oil_temperature`, `Motor_current`, `DV_eletric`, `Towers`, `month`, `day`

### 5. Modeling
- Model: **Random Forest Classifier** with `class_weight='balanced'`
- Custom decision threshold (default 0.5 catches zero failures → lowered to 0.15)
- Evaluated only on **Class 1 metrics** (Recall, Precision, F1) — accuracy is misleading with imbalanced data

### 6. Hyperparameter Optimization
- Compared **RandomizedSearchCV** (random blind search) vs **Optuna** (Bayesian smart search)
- Used Optuna with 30 trials on the reduced dataset
- Best params: `sampling_strategy=0.43`, `n_estimators=358`, `max_depth=6`, `threshold=0.06`

---

## 📊 Main Findings

- **SMOTE (sampling_strategy=0.05)** achieved the best balance between Recall and F1, outperforming undersampling strategies
- **DV_pressure**, **Oil_temperature**, **TP2**, and **Motor_current** are the most important features for failure detection
- The default threshold of **0.5 fails completely** — the model's max predicted probability for failures is ~0.42, so threshold tuning is essential
- **ROC AUC is misleading** for this dataset — the model scores 0.997 because 98% of data is class 0; **F1 and Recall are the real metrics**
- The model detects failures **as they happen**, not before — redefining the target as a pre-failure window is the key next step

| Experiment | Recall | F1 Score | Precision |
|---|---|---|---|
| Baseline | 0.495 | 0.508 | 0.523 |
| Undersample 50/50 | 0.737 | 0.398 | 0.273 |
| Undersample 70/30 | 0.736 | 0.428 | 0.302 |
| **SMOTE 0.05 ⭐** | **0.691** | **0.634** | **0.586** |
| SMOTETomek 0.05 | 0.624 | 0.608 | 0.593 |
| Optuna (threshold=0.06) | 0.715 | 0.334 | 0.218 |

---

## 🚀 Next Steps / Ideas for Improvement

- **Redefine the target variable** — mark the 1 week before each failure as class 1, enabling true *predictive* maintenance (predict before, not during)
- **Rolling window features** — add `TP2_rolling_mean`, `Oil_temperature_rolling_std` to capture gradual sensor drift
- **Try XGBoost / LightGBM** — gradient boosting models handle imbalance natively with `scale_pos_weight`
- **Deep Learning** — LSTM or TabNet to capture long-term time series dependencies before failure
- **Retrain on full resolution data** — 15M rows at 1Hz instead of the 1-minute resampled version
- **Use Average Precision (PR AUC)** as primary metric instead of ROC AUC

---

## 📁 Repo Structure

```
Predictive-Maintenance-for-Railway-Aircompressor-MetroPT-3/
│
├── 1_EDA.ipynb                  # Exploratory Data Analysis
├── 2_baseline.ipynb             # Baseline model + resampling experiments
├── 3_random_forest.ipynb        # Feature engineering + Optuna optimization
├── utils.py
│
├── results/                     # All experiment results tracked here
│   └── metrics.csv              
|   ├── confusion_matrices.csv 
│   └── ...
|
├── images/
│   ├── class-imbalance.png
│   ├── correlation.png
│   ├── feature_importance.png
│   ├── metrics.png
│   └── ...
│
├── environment.yml              # Conda environment
├── requirements.txt             # Pip requirements
└── README.md
