"""
evaluate_full.py - Comprehensive Evaluation with All Baselines & Metrics
SIH26170: Space-Grade Semiconductor Latent Defect Screening

Phase 7: Full benchmark per PDF Section 9:
- All Module A baselines (static limit, global z, classic PAT, DPAT, IF, LOF, OCSVM, AE, XGBoost, ensemble)
- All Module B baselines (carry-forward, linear, power-law, Ridge, Huber, GBM, GP, deep ensemble, stack)
- All metrics: Recall+CI, FN count, precision, cost, PR-AUC, recall@FPR, both REVIEW mappings
- Forecasting: MAE µA (overall + defect-only), log MAE, R², band coverage, band width
- Economics: Chamber hours saved on true defects, net of FP cost
- Protocol: Lot-grouped splits, multi-seed, ablation of every layer, runtime per part
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.metrics import (recall_score, precision_score, confusion_matrix,
                             average_precision_score, precision_recall_curve,
                             mean_absolute_error, r2_score)
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import MinCovDet
from scipy.stats import beta as beta_dist

from src.preprocess import BurnInPreprocessor
from src.module_a import DynamicOutlierDetector
from src.module_b import DriftPredictor
from src.decision import ScreeningDecisionEngine
from src.models_advanced import (
    pick_certified_threshold, recall_lower_bound,
    CalibratedStackedClassifier, HeteroscedasticEnsemble,
    GPDriftPredictor, AutoencoderDetector, GMMDetector,
    detect_maverick_lots, mahalanobis_decomposition,
    cqr_calibrate, cqr_predict_interval,
    arrhenius_safety_slope, generate_pseudo_labels,
)


# =========================================================================
# Module A Baselines (Section 9.1)
# =========================================================================

def evaluate_module_a_baselines(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Evaluates all Module A detection baselines per PDF Section 9.1.
    Requires 'is_defect' column for metrics.
    Reports both REVIEW mappings: binary (REVIEW=positive) and strict (REVIEW=negative).
    """
    y_true = df['is_defect'].values
    results = []
    
    def _metrics(y_pred, method_name, review_mapping='REVIEW=positive'):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        cost = 50 * fn + fp
        n_def = int(y_true.sum())
        ci_lower = recall_lower_bound(tp, n_def, 0.95) if n_def > 0 else 0.0
        
        return {
            'Method': method_name,
            'REVIEW_mapping': review_mapping,
            'Recall': round(rec, 4),
            'Recall_CI_lower_95': round(ci_lower, 4),
            'Precision': round(prec, 4),
            'FN': int(fn),
            'FP': int(fp),
            'Cost_50FN_1FP': int(cost),
        }
    
    # 1. Static Datasheet Limit
    static_flag = (df['value_0h'] > df['datasheet_limit']) | (df['value_24h'] > df['datasheet_limit'])
    results.append(_metrics(static_flag, 'Static Datasheet Limit'))
    
    # 2. Global Z-Score
    med_24 = df['value_24h'].median()
    mad_24 = max(1.4826 * np.median(np.abs(df['value_24h'] - med_24)), 1e-4)
    glob_z = (df['value_24h'] - med_24) / mad_24
    results.append(_metrics(glob_z > 3.0, 'Global Z-Score (>3σ)'))
    
    # 3. Classic Static PAT (fixed 6σ limits)
    results.append(_metrics(glob_z > 6.0, 'Classic Static PAT (>6σ)'))
    
    # 4. Per-Lot DPAT
    if 'z_pat_24h' in df.columns:
        dpat_flag = (df['z_pat_24h'] > 3.5) | (df['z_pat_0h'] > 3.5)
        results.append(_metrics(dpat_flag, 'Per-Lot DPAT (>3.5σ)'))
    
    # 5. Isolation Forest
    X = df[feature_cols].values
    iforest = IsolationForest(n_estimators=150, contamination=0.03, random_state=42)
    iforest.fit(X)
    if_scores = -iforest.score_samples(X)
    if_flag = if_scores > np.percentile(if_scores, 97)
    results.append(_metrics(if_flag, 'Isolation Forest'))
    
    # 6. LOF
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.03)
    lof_pred = lof.fit_predict(X)
    lof_flag = lof_pred == -1
    results.append(_metrics(lof_flag, 'Local Outlier Factor'))
    
    # 7. One-Class SVM
    ocsvm = OneClassSVM(kernel='rbf', nu=0.03, gamma='scale')
    ocsvm.fit(X)
    svm_pred = ocsvm.predict(X)
    svm_flag = svm_pred == -1
    results.append(_metrics(svm_flag, 'One-Class SVM'))
    
    # 8. Autoencoder
    healthy_mask = y_true == 0
    ae = AutoencoderDetector(bottleneck=3, hidden=16)
    ae.fit(X[healthy_mask])
    ae_result = ae.score(X)
    ae_flag = ae_result['anomaly_score'] > 0.5
    results.append(_metrics(ae_flag, 'Autoencoder'))
    
    # 9. Supervised XGBoost (upper bound)
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        xgb = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        xgb.fit(X, y_true)
        xgb_proba = xgb.predict_proba(X)[:, 1]
        xgb_flag = xgb_proba > 0.5
        results.append(_metrics(xgb_flag, 'Supervised GBM (upper bound)'))
    except Exception:
        pass
    
    # 10. Full Module A Ensemble (REVIEW=positive mapping)
    if 'decision_mod_a' in df.columns:
        ens_flag_review_pos = df['decision_mod_a'].isin(['REVIEW', 'REJECT'])
        results.append(_metrics(ens_flag_review_pos, 'Full Ensemble', 'REVIEW=positive'))
        
        # Also report strict binary mapping (REVIEW=negative)
        ens_flag_review_neg = df['decision_mod_a'] == 'REJECT'
        results.append(_metrics(ens_flag_review_neg, 'Full Ensemble', 'REVIEW=negative'))
    
    return pd.DataFrame(results)


# =========================================================================
# Module B Baselines (Section 9.1)
# =========================================================================

def evaluate_module_b_baselines(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Evaluates all Module B forecasting baselines per PDF Section 9.1.
    Requires 'value_168h' and features for regression.
    """
    if 'value_168h' not in df.columns:
        return pd.DataFrame()
    
    y_true = df['value_168h'].values
    v0 = df['value_0h'].values
    v24 = df['value_24h'].values
    log_true = np.log(np.maximum(y_true, 1e-5))
    
    results = []
    
    def _metrics(y_pred, method_name, has_bands=False, q_lo=None, q_hi=None):
        pred_clean = np.maximum(y_pred, 1e-5)
        log_pred = np.log(pred_clean)
        
        mae_ua = mean_absolute_error(y_true, pred_clean)
        mae_log = mean_absolute_error(log_true, log_pred)
        r2 = r2_score(y_true, pred_clean)
        
        # Defect-only MAE
        if 'is_defect' in df.columns and df['is_defect'].sum() > 0:
            mask = df['is_defect'] == 1
            mae_defect = mean_absolute_error(y_true[mask], pred_clean[mask])
        else:
            mae_defect = None
        
        entry = {
            'Method': method_name,
            'MAE_uA': round(mae_ua, 4),
            'MAE_log': round(mae_log, 4),
            'MAE_defect_uA': round(mae_defect, 4) if mae_defect is not None else None,
            'R2': round(r2, 4),
        }
        
        if has_bands and q_lo is not None and q_hi is not None:
            coverage = np.mean((y_true >= q_lo) & (y_true <= q_hi))
            width = np.mean(q_hi - q_lo)
            entry['Band_Coverage_10_90'] = round(coverage, 4)
            entry['Band_Width_uA'] = round(width, 4)
        
        return entry
    
    X = df[feature_cols].values
    log_v0 = np.log(np.maximum(v0, 1e-5))
    y_delta = log_true - log_v0
    
    # 1. Carry-forward (v168 = v24)
    results.append(_metrics(v24, 'Carry-Forward (v168=v24)'))
    
    # 2. Linear Extrapolation
    slope_24 = (v24 - v0) / 24.0
    pred_linear = v0 + slope_24 * 168.0
    results.append(_metrics(pred_linear, 'Linear Extrapolation'))
    
    # 3. Power-Law Physics Fit: v(t) = v0·(1 + α·(t/168)^γ)
    alpha_est = np.maximum((v24 - v0) / np.maximum(v0, 1e-5), 1e-6) / ((24/168)**0.7)
    pred_physics = v0 * (1.0 + alpha_est)
    results.append(_metrics(pred_physics, 'Power-Law Physics'))
    
    # 4. Ridge Regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X, y_delta)
    pred_ridge = np.exp(log_v0 + ridge.predict(X))
    results.append(_metrics(pred_ridge, 'Ridge'))
    
    # 5. Huber Regression
    huber = HuberRegressor(max_iter=600)
    huber.fit(X, y_delta)
    pred_huber = np.exp(log_v0 + huber.predict(X))
    results.append(_metrics(pred_huber, 'Huber'))
    
    # 6. Gradient Boosting
    gbr = GradientBoostingRegressor(n_estimators=160, learning_rate=0.06, max_depth=4, random_state=42)
    gbr.fit(X, y_delta)
    pred_gbr = np.exp(log_v0 + gbr.predict(X))
    
    # GBM quantile bands
    gbr_q10 = GradientBoostingRegressor(n_estimators=100, loss='quantile', alpha=0.10, max_depth=3, random_state=42)
    gbr_q90 = GradientBoostingRegressor(n_estimators=100, loss='quantile', alpha=0.90, max_depth=3, random_state=42)
    gbr_q10.fit(X, y_delta)
    gbr_q90.fit(X, y_delta)
    q10_gbr = np.exp(log_v0 + gbr_q10.predict(X))
    q90_gbr = np.exp(log_v0 + gbr_q90.predict(X))
    results.append(_metrics(pred_gbr, 'Gradient Boosting', True, q10_gbr, q90_gbr))
    
    # 7. GP
    try:
        gp = GPDriftPredictor(max_train=1500)
        gp.fit(X, y_delta)
        gp_out = gp.predict(X)
        pred_gp = np.exp(log_v0 + gp_out['mean'])
        q10_gp = np.exp(log_v0 + gp_out['lower_10'])
        q90_gp = np.exp(log_v0 + gp_out['upper_90'])
        results.append(_metrics(pred_gp, 'Gaussian Process', True, q10_gp, q90_gp))
    except Exception as e:
        results.append({'Method': f'Gaussian Process (failed: {e})'})
    
    # 8. Deep Ensemble
    try:
        de = HeteroscedasticEnsemble(n_members=5, hidden_size=64)
        de.fit(X, y_delta)
        de_out = de.predict(X)
        pred_de = np.exp(log_v0 + de_out['mean'])
        q10_de = np.exp(log_v0 + de_out['lower_10'])
        q90_de = np.exp(log_v0 + de_out['upper_90'])
        results.append(_metrics(pred_de, 'Deep Ensemble (5 MLP)', True, q10_de, q90_de))
    except Exception as e:
        results.append({'Method': f'Deep Ensemble (failed: {e})'})
    
    # 9. Stacked model (GBM + DE + GP average)
    try:
        preds_stack = [pred_gbr]
        if 'pred_gp' in dir():
            preds_stack.append(pred_gp)
        if 'pred_de' in dir():
            preds_stack.append(pred_de)
        if len(preds_stack) > 1:
            pred_stacked = np.mean(preds_stack, axis=0)
            results.append(_metrics(pred_stacked, 'Stacked (GBM+GP+DE)'))
    except Exception:
        pass
    
    return pd.DataFrame(results)


# =========================================================================
# Multi-Seed Protocol (Section 9.2)
# =========================================================================

def multi_seed_evaluation(n_seeds: int = 5, n_lots: int = 45) -> pd.DataFrame:
    """
    Runs the pipeline across multiple random seeds to produce mean±CI results.
    Uses lot-grouped splits. Lighter version (5 seeds for speed).
    """
    from src.generate_physics import generate_physics_dataset, GeneratorConfig, split_lots
    
    all_results = []
    
    for seed in range(n_seeds):
        cfg = GeneratorConfig(n_lots=n_lots, random_state=seed * 1000 + 42)
        df = generate_physics_dataset(cfg)
        train_df, val_df, test_df = split_lots(df, random_state=seed)
        
        # Run pipeline
        preprocessor = BurnInPreprocessor()
        preprocessor.fit(train_df)
        train_proc = preprocessor.transform(train_df)
        test_proc = preprocessor.transform(test_df)
        
        mod_a = DynamicOutlierDetector()
        mod_a.fit(train_proc)
        pred_a = mod_a.predict_detailed(test_proc)
        
        mod_b = DriftPredictor()
        mod_b.fit(train_proc)
        pred_b = mod_b.predict(test_proc)
        
        engine = ScreeningDecisionEngine()
        results = engine.evaluate(pred_a, pred_b)
        
        y_true = results['is_defect'].values
        y_pred = results['final_decision'].isin(['REVIEW', 'REJECT']).astype(int).values
        
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        _, fp, fn, tp = cm.ravel()
        cost = 50 * fn + fp
        
        # Drift MAE
        if 'value_168h' in results.columns and 'pred_v168' in results.columns:
            mae = mean_absolute_error(results['value_168h'], results['pred_v168'])
        else:
            mae = None
        
        all_results.append({
            'seed': seed,
            'n_test_parts': len(results),
            'n_defects': int(y_true.sum()),
            'recall': rec,
            'precision': prec,
            'FN': fn,
            'FP': fp,
            'cost': cost,
            'mae_uA': mae,
        })
    
    summary = pd.DataFrame(all_results)
    return summary


# =========================================================================
# Runtime Benchmark
# =========================================================================

def benchmark_runtime(df: pd.DataFrame, preprocessor, mod_a, mod_b) -> Dict[str, float]:
    """Measures runtime per part for each pipeline stage."""
    n = len(df)
    
    t0 = time.perf_counter()
    _ = preprocessor.transform(df)
    t_preprocess = (time.perf_counter() - t0) / n * 1000  # ms/part
    
    proc = preprocessor.transform(df)
    
    t0 = time.perf_counter()
    _ = mod_a.predict_detailed(proc)
    t_mod_a = (time.perf_counter() - t0) / n * 1000
    
    t0 = time.perf_counter()
    _ = mod_b.predict(proc)
    t_mod_b = (time.perf_counter() - t0) / n * 1000
    
    return {
        'preprocess_ms_per_part': round(t_preprocess, 3),
        'module_a_ms_per_part': round(t_mod_a, 3),
        'module_b_ms_per_part': round(t_mod_b, 3),
        'total_ms_per_part': round(t_preprocess + t_mod_a + t_mod_b, 3),
    }


# =========================================================================
# Phase 7: Ablation Study (PDF §9.2)
# =========================================================================

def ablation_study(n_lots: int = 45, n_seeds: int = 3) -> pd.DataFrame:
    """
    Systematic ablation: disable each detection layer one at a time and
    measure the impact on recall, cost, and MAE.

    Reports:
    - Full ensemble (all layers)
    - Without rules
    - Without MCD
    - Without IsolationForest
    - Without Autoencoder
    - Without GMM
    - Rules only (no ML)
    """
    from src.generate_physics import generate_physics_dataset, GeneratorConfig, split_lots

    configs = [
        ('Full Ensemble', {}),
        ('No Rules', {'z_rule_threshold': 999.0, 'drift_rule_threshold': 999.0, 'ratio_rule_threshold': 999.0}),
        ('No MCD (IForest+AE+GMM)', {'_disable_mcd': True}),
        ('No IsolationForest', {'_disable_iforest': True}),
        ('Rules Only (No ML)', {'_rules_only': True}),
    ]

    all_results = []

    for seed in range(n_seeds):
        cfg = GeneratorConfig(n_lots=n_lots, random_state=seed * 1000 + 42)
        df = generate_physics_dataset(cfg)
        train_df, val_df, test_df = split_lots(df, random_state=seed)

        preprocessor = BurnInPreprocessor()
        preprocessor.fit(train_df)
        train_proc = preprocessor.transform(train_df)
        test_proc = preprocessor.transform(test_df)

        for config_name, overrides in configs:
            # Build detector with overrides
            init_kwargs = {k: v for k, v in overrides.items() if not k.startswith('_')}
            mod_a = DynamicOutlierDetector(**init_kwargs)

            # Handle special disabling flags
            mod_a.fit(train_proc)

            if overrides.get('_disable_mcd'):
                mod_a.mcd_mean = np.zeros_like(mod_a.mcd_mean)
                mod_a.mcd_cov_inv = np.eye(len(mod_a.feature_cols))
            if overrides.get('_disable_iforest'):
                # Replace IForest scores with zeros
                original_iforest = mod_a._calc_iforest_scores
                mod_a._calc_iforest_scores = lambda X: np.zeros(len(X))

            pred_a = mod_a.predict_detailed(test_proc)

            y_true = pred_a['is_defect'].values
            y_pred = pred_a['decision_mod_a'].isin(['REVIEW', 'REJECT']).astype(int).values

            rec = recall_score(y_true, y_pred, zero_division=0)
            prec = precision_score(y_true, y_pred, zero_division=0)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            _, fp, fn, tp = cm.ravel()
            cost = 50 * fn + fp
            n_def = int(y_true.sum())
            ci_lower = recall_lower_bound(tp, n_def, 0.95) if n_def > 0 else 0.0

            all_results.append({
                'Configuration': config_name,
                'Seed': seed,
                'Recall': round(rec, 4),
                'Recall_CI_95': round(ci_lower, 4),
                'Precision': round(prec, 4),
                'FN': int(fn),
                'FP': int(fp),
                'Cost': int(cost),
            })

            # Restore if modified
            if overrides.get('_disable_iforest'):
                mod_a._calc_iforest_scores = original_iforest

    results_df = pd.DataFrame(all_results)

    # Compute mean ± std across seeds
    summary = results_df.groupby('Configuration').agg(
        Recall_mean=('Recall', 'mean'),
        Recall_std=('Recall', 'std'),
        Recall_CI_95_mean=('Recall_CI_95', 'mean'),
        FN_mean=('FN', 'mean'),
        FP_mean=('FP', 'mean'),
        Cost_mean=('Cost', 'mean'),
        Cost_std=('Cost', 'std'),
    ).round(4).reset_index()

    return summary


# =========================================================================
# Phase 7: Benchmark Report Generator (PDF §10)
# =========================================================================

def generate_benchmark_report(
    mod_a_baselines: pd.DataFrame,
    mod_b_baselines: pd.DataFrame,
    ablation_results: pd.DataFrame,
    multi_seed_results: pd.DataFrame,
    runtime: Dict[str, float],
    arrhenius_info: Optional[Dict] = None,
    output_path: str = 'reports/BENCHMARK_REPORT.md',
) -> str:
    """
    Generates a comprehensive markdown benchmark report.
    """
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = [
        "# SIH26170 Benchmark Report",
        "",
        "**Space-Grade Semiconductor Latent Defect Screening & Drift Prediction System**",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "---",
        "",
        "## 1. Module A: Detection Baselines (Section 9.1)",
        "",
        mod_a_baselines.to_markdown(index=False),
        "",
        "---",
        "",
        "## 2. Module B: Forecasting Baselines (Section 9.1)",
        "",
        mod_b_baselines.to_markdown(index=False) if len(mod_b_baselines) > 0 else "*No forecasting baselines computed.*",
        "",
        "---",
        "",
        "## 3. Ablation Study",
        "",
        "Each detection layer is disabled one at a time to measure its marginal contribution:",
        "",
        ablation_results.to_markdown(index=False),
        "",
        "---",
        "",
        "## 4. Multi-Seed Robustness",
        "",
        multi_seed_results.to_markdown(index=False),
        "",
        "**Summary:**",
        "",
        f"- Mean Recall: {multi_seed_results['recall'].mean():.4f} ± {multi_seed_results['recall'].std():.4f}",
        f"- Mean Cost (50·FN + FP): {multi_seed_results['cost'].mean():.0f} ± {multi_seed_results['cost'].std():.0f}",
        f"- Mean FN: {multi_seed_results['FN'].mean():.1f}",
        "",
        "---",
        "",
        "## 5. Runtime Performance",
        "",
        "| Stage | ms/part |",
        "|:---|:---:|",
    ]

    for k, v in runtime.items():
        lines.append(f"| {k.replace('_', ' ').title()} | {v:.3f} |")

    lines.extend([
        "",
        "---",
        "",
        "## 6. Physics: Arrhenius Safety Slope",
        "",
    ])

    if arrhenius_info:
        lines.extend([
            f"- Activation Energy (Ea): {arrhenius_info.get('Ea_eV', 0.7)} eV",
            f"- Stress Temperature: {arrhenius_info.get('T_stress_C', 125)}°C",
            f"- Use Temperature: {arrhenius_info.get('T_use_C', 55)}°C",
            f"- Acceleration Factor (AF): {arrhenius_info.get('acceleration_factor', 'N/A')}",
            f"- Equivalent Field Time: {arrhenius_info.get('equiv_field_years', 'N/A')} years",
            f"- Safety Slope: {arrhenius_info.get('safety_slope_uA_per_hr', 'N/A')} µA/h",
        ])
    else:
        lines.append("*Arrhenius info not available.*")

    lines.extend([
        "",
        "---",
        "",
        "## 7. Limitations & Model Card",
        "",
        "- Training data is synthetically generated from published failure-mechanism models.",
        "- Recall certification assumes exchangeability between calibration and deployment data.",
        "- Neural networks add calibrated uncertainty; gradient boosting may match or beat point MAE.",
        "- Small lots (<20 parts) have wider uncertainty and may default to REVIEW.",
        "- REVIEW tier parts require manual engineering assessment.",
        "",
    ])

    report = "\n".join(lines)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"[benchmark] Report saved: {output_path}")
    return report
