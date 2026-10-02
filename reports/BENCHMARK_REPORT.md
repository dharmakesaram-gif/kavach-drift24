# SIH26170 Benchmark Report

**Space-Grade Semiconductor Latent Defect Screening & Drift Prediction System**

Generated: 2026-10-02 19:34:23

---

## 1. Module A: Detection Baselines (Section 9.1)

| Method                               | Recall (Catch Rate)   | Precision   |   False Negatives (Catastrophic) |   False Positives (Scrap) |   Cost Score (50*FN + 1*FP) |
|:-------------------------------------|:----------------------|:------------|---------------------------------:|--------------------------:|----------------------------:|
| Static Datasheet Limit Only          | 0.0%                  | 0.0%        |                               39 |                         0 |                        1950 |
| Global Z-Score (No Lot Awareness)    | 43.6%                 | 81.0%       |                               22 |                         4 |                        1104 |
| Module A: Dynamic Lot-Aware Ensemble | 74.4%                 | 65.9%       |                               10 |                        15 |                         515 |

---

## 2. Module B: Forecasting Baselines (Section 9.1)

|   mae_original_scale_uA |   mae_log_scale | mae_huber_baseline_uA   |   r2_score |   early_rejections_at_24h |   true_defect_rejections |   false_positive_rejections |   chamber_hours_saved |   fp_chamber_cost_hours |   mae_defects_only |
|------------------------:|----------------:|:------------------------|-----------:|--------------------------:|-------------------------:|----------------------------:|----------------------:|------------------------:|-------------------:|
|                  0.2267 |          0.0172 |                         |     0.9134 |                        50 |                       37 |                          13 |                  5328 |                    1872 |             4.3255 |

---

## 3. Ablation Study

Each detection layer is disabled one at a time to measure its marginal contribution:

| Configuration           |   Recall_mean |   Recall_std |   Recall_CI_95_mean |   FN_mean |   FP_mean |   Cost_mean |   Cost_std |
|:------------------------|--------------:|-------------:|--------------------:|----------:|----------:|------------:|-----------:|
| Full Ensemble           |        0.7693 |       0.0795 |              0.609  |         7 |    6      |     356     |    150.01  |
| No IsolationForest      |        0.7693 |       0.0795 |              0.609  |         7 |    5.6667 |     355.667 |    150.018 |
| No MCD (IForest+AE+GMM) |        0.7693 |       0.0795 |              0.609  |         7 |    5.6667 |     355.667 |    150.018 |
| No Rules                |        0.6659 |       0.1236 |              0.5013 |        10 |    0.3333 |     500.333 |    217.601 |
| Rules Only (No ML)      |        0.7693 |       0.0795 |              0.609  |         7 |    6      |     356     |    150.01  |

---

## 4. Multi-Seed Robustness

|   recall |   cost |   FN |
|---------:|-------:|-----:|
|   0.7693 |    356 |    7 |

**Summary:**

- Mean Recall: 0.7693 ± nan
- Mean Cost (50·FN + FP): 356 ± nan
- Mean FN: 7.0

---

## 5. Runtime Performance

| Stage | ms/part |
|:---|:---:|
| Preprocess Ms Per Part | 0.017 |
| Module A Ms Per Part | 0.232 |
| Module B Ms Per Part | 0.014 |
| Total Ms Per Part | 0.263 |

---

## 6. Physics: Arrhenius Safety Slope

- Activation Energy (Ea): 0.7 eV
- Stress Temperature: 125.0°C
- Use Temperature: 55.0°C
- Acceleration Factor (AF): 77.7
- Equivalent Field Time: 1.49 years
- Safety Slope: 0.034617 µA/h

---

## 7. Limitations & Model Card

- Training data is synthetically generated from published failure-mechanism models.
- Recall certification assumes exchangeability between calibration and deployment data.
- Neural networks add calibrated uncertainty; gradient boosting may match or beat point MAE.
- Small lots (<20 parts) have wider uncertainty and may default to REVIEW.
- REVIEW tier parts require manual engineering assessment.
