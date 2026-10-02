"""
evaluate.py - End-to-End Pipeline Evaluation, Baselines & Stress Testing
SIH26170: Space-Grade Semiconductor Latent Defect Screening

Executes:
1. End-to-End Pipeline on unseen test lots
2. Rigorous Baselines Comparison (Static Limits vs Global Z vs Dynamic Ensemble)
3. Cost-Weighted Score (Cost = 50 * FN + 1 * FP) & Recall >= 99% verification
4. Drift Prediction MAE (physical µA & log-space)
5. Layer Ablation Study
6. Generates Comprehensive Benchmark Report in reports/
"""

import os
import json
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Any

from src.generate import generate_burnin_dataset, split_lots
from src.preprocess import BurnInPreprocessor
from src.module_a import DynamicOutlierDetector, evaluate_baselines
from src.module_b import DriftPredictor, evaluate_drift_prediction
from src.decision import ScreeningDecisionEngine
from src.explain import generate_qa_report_card


def run_full_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Executes training, validation calibration, and testing of Module A + Module B + Decision Engine.
    """
    # 1. Preprocessing
    preprocessor = BurnInPreprocessor(small_lot_threshold=30, shrinkage_prior_weight=15.0)
    preprocessor.fit(train_df)
    
    train_proc = preprocessor.transform(train_df)
    val_proc = preprocessor.transform(val_df)
    test_proc = preprocessor.transform(test_df)
    
    # 2. Module A: Outlier Detector
    mod_a = DynamicOutlierDetector(cost_fn_weight=50.0, cost_fp_weight=1.0)
    mod_a.fit(train_proc, val_proc)
    
    # 3. Module B: Drift Predictor
    mod_b = DriftPredictor(safety_k_sigma=3.0)
    mod_b.fit(train_proc)
    
    # 4. Predict on Test Set (Unseen Lots)
    test_pred_a = mod_a.predict_detailed(test_proc)
    test_pred_b = mod_b.predict(test_proc)
    
    # 5. Combined Decision Engine
    decision_engine = ScreeningDecisionEngine()
    test_final = decision_engine.evaluate(test_pred_a, test_pred_b)
    
    # 6. Evaluate Module A Baselines
    baseline_comparison_df = evaluate_baselines(test_final)
    
    # 7. Evaluate Module B Drift Accuracy
    drift_metrics = evaluate_drift_prediction(test_final)
    
    # 8. Overall System Metrics on Test Lots
    y_true = test_final['is_defect'].values
    y_pred = (test_final['final_decision'].isin(['REVIEW', 'REJECT'])).astype(int).values
    
    from sklearn.metrics import recall_score, precision_score, confusion_matrix
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    cost = 50 * fn + 1 * fp
    
    # Defect Type Breakdown (Checking recall on each defect mechanism)
    defect_type_stats = {}
    for dtype, group in test_final.groupby('defect_type'):
        d_true = group['is_defect'].values
        d_pred = (group['final_decision'].isin(['REVIEW', 'REJECT'])).astype(int).values
        if d_true.sum() > 0:
            d_rec = recall_score(d_true, d_pred, zero_division=0)
            defect_type_stats[dtype] = {
                'count': len(group),
                'caught': int(d_pred.sum()),
                'recall': round(float(d_rec), 4)
            }
        else:
            defect_type_stats[dtype] = {
                'count': len(group),
                'passed': int((group['final_decision'] == 'ACCEPT').sum()),
                'false_alarm_rate': round(float(d_pred.sum() / len(group)), 4)
            }
            
    # Sample QA Report Cards for 2 flagged parts
    flagged_parts = test_final[test_final['final_decision'] == 'REJECT']
    sample_reports = []
    if len(flagged_parts) > 0:
        for _, r in flagged_parts.head(2).iterrows():
            sample_reports.append(generate_qa_report_card(r))
            
    summary_results = {
        'test_lot_count': test_final['lot_id'].nunique(),
        'test_part_count': len(test_final),
        'total_defects_in_test': int(y_true.sum()),
        'defects_caught': int(tp),
        'false_negatives_escapes': int(fn),
        'false_positives_overkill': int(fp),
        'system_recall': round(float(rec), 4),
        'system_precision': round(float(prec), 4),
        'cost_score_50FN_1FP': int(cost),
        'drift_metrics': drift_metrics,
        'defect_type_breakdown': defect_type_stats,
        'baseline_comparison': baseline_comparison_df.to_dict(orient='records'),
        'sample_reports': sample_reports
    }
    
    return {
        'summary': summary_results,
        'baseline_df': baseline_comparison_df,
        'test_results_df': test_final,
        'preprocessor': preprocessor,
        'mod_a': mod_a,
        'mod_b': mod_b
    }


def run_ablation_study(train_proc: pd.DataFrame, test_proc: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluates individual contribution of each detector layer:
    1. Rules Only
    2. Mahalanobis Only
    3. Isolation Forest Only
    4. Full Multi-Layer Ensemble
    """
    from sklearn.metrics import recall_score, precision_score, confusion_matrix
    y_true = test_proc['is_defect'].values
    
    # 1. Rules only
    z_rule = (test_proc['z_pat_0h'] > 3.5) | (test_proc['z_pat_24h'] > 3.5) | (test_proc['z_drift_24h'] > 3.2)
    rec_r = recall_score(y_true, z_rule, zero_division=0)
    cm_r = confusion_matrix(y_true, z_rule, labels=[0, 1])
    cost_r = 50 * cm_r[1, 0] + cm_r[0, 1]
    
    # 2. Mahalanobis only
    mcd = DynamicOutlierDetector()
    mcd.fit(train_proc)
    res_m = mcd.predict_detailed(test_proc)
    pred_mahal = res_m['score_mahalanobis'] >= 0.50
    rec_m = recall_score(y_true, pred_mahal, zero_division=0)
    cm_m = confusion_matrix(y_true, pred_mahal, labels=[0, 1])
    cost_m = 50 * cm_m[1, 0] + cm_m[0, 1]
    
    # 3. Isolation Forest only
    pred_iforest = res_m['score_iforest'] >= 0.50
    rec_if = recall_score(y_true, pred_iforest, zero_division=0)
    cm_if = confusion_matrix(y_true, pred_iforest, labels=[0, 1])
    cost_if = 50 * cm_if[1, 0] + cm_if[0, 1]
    
    # 4. Full Ensemble
    pred_ens = res_m['decision_mod_a'].isin(['REVIEW', 'REJECT'])
    rec_ens = recall_score(y_true, pred_ens, zero_division=0)
    cm_ens = confusion_matrix(y_true, pred_ens, labels=[0, 1])
    cost_ens = 50 * cm_ens[1, 0] + cm_ens[0, 1]
    
    ablation_data = [
        {'Layer': '1. AEC-Q001 Rules Only', 'Recall': f"{rec_r*100:.1f}%", 'Escapes (FN)': cm_r[1, 0], 'Cost Score': cost_r},
        {'Layer': '2. Robust Mahalanobis Only', 'Recall': f"{rec_m*100:.1f}%", 'Escapes (FN)': cm_m[1, 0], 'Cost Score': cost_m},
        {'Layer': '3. Isolation Forest Only', 'Recall': f"{rec_if*100:.1f}%", 'Escapes (FN)': cm_if[1, 0], 'Cost Score': cost_if},
        {'Layer': '4. Full Layered Ensemble', 'Recall': f"{rec_ens*100:.1f}%", 'Escapes (FN)': cm_ens[1, 0], 'Cost Score': cost_ens}
    ]
    return pd.DataFrame(ablation_data)


def df_to_markdown(df: pd.DataFrame) -> str:
    """Safely converts DataFrame to Markdown table without external tabulate dependency."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in r) + " |")
    return "\n".join([header, sep] + rows)


if __name__ == '__main__':
    data_dir = "data/synthetic"
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    train_path = os.path.join(data_dir, "burnin_train.csv")
    val_path = os.path.join(data_dir, "burnin_val.csv")
    test_path = os.path.join(data_dir, "burnin_test.csv")
    
    # Generate data if not already existing
    if not os.path.exists(train_path):
        print("Data files not found. Generating synthetic dataset...")
        df = generate_burnin_dataset(n_lots=45, random_state=42)
        train_df, val_df, test_df = split_lots(df)
        os.makedirs(data_dir, exist_ok=True)
        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
    else:
        print("Loading existing dataset files...")
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        test_df = pd.read_csv(test_path)
        
    print(f"Running pipeline on {len(test_df)} test parts ({test_df['lot_id'].nunique()} unseen lots)...")
    results = run_full_pipeline(train_df, val_df, test_df)
    
    # Save test results with predictions
    test_results_path = os.path.join(reports_dir, "test_predictions_detailed.csv")
    results['test_results_df'].to_csv(test_results_path, index=False)
    
    # Save baseline comparison
    baseline_path = os.path.join(reports_dir, "baseline_comparison.csv")
    results['baseline_df'].to_csv(baseline_path, index=False)
    
    # Run and save Ablation Study
    train_proc = results['preprocessor'].transform(train_df)
    test_proc = results['preprocessor'].transform(test_df)
    ablation_df = run_ablation_study(train_proc, test_proc)
    ablation_path = os.path.join(reports_dir, "ablation_study.csv")
    ablation_df.to_csv(ablation_path, index=False)
    
    # Save JSON summary
    summary_path = os.path.join(reports_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results['summary'], f, indent=2)
        
    # Generate Markdown Benchmark Report
    report_md_path = os.path.join(reports_dir, "BENCHMARK_REPORT.md")
    with open(report_md_path, 'w', encoding='utf-8') as f:
        f.write("# SIH26170: Burn-In Latent Defect Detection Benchmark Report\n\n")
        f.write(f"**Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. Executive Summary\n\n")
        f.write(f"- **Defect Recall (Interception Rate):** **{results['summary']['system_recall']*100:.2f}%** (Target: ≥ 99.0%)\n")
        f.write(f"- **Catastrophic Field Escapes (False Negatives):** **{results['summary']['false_negatives_escapes']}** (Target: 0 Escapes)\n")
        f.write(f"- **Cost-Weighted Penalty (50×FN + 1×FP):** **{results['summary']['cost_score_50FN_1FP']}**\n")
        f.write(f"- **Drift Prediction Accuracy MAE (168h):** **{results['summary']['drift_metrics']['mae_original_scale_uA']:.4f} µA** (Log MAE: {results['summary']['drift_metrics']['mae_log_scale']:.4f})\n")
        f.write(f"- **Burn-In Thermal Chamber Hours Saved:** **{results['summary']['drift_metrics']['chamber_hours_saved']:,} hours** at 24h early termination\n\n")
        f.write("## 2. Industry Baseline Comparison\n\n")
        f.write(df_to_markdown(results['baseline_df']) + "\n\n")
        f.write("## 3. Analytical Layer Ablation Study\n\n")
        f.write(df_to_markdown(ablation_df) + "\n\n")
        f.write("## 4. Defect Mechanism Interception Breakdown\n\n")
        d_df = pd.DataFrame(results['summary']['defect_type_breakdown']).T.reset_index()
        d_df.columns = ['Defect Type'] + list(d_df.columns[1:])
        f.write(df_to_markdown(d_df) + "\n\n")
        
    print("\n" + "="*70)
    print("           SIH26170 PIPELINE BENCHMARK SUMMARY")
    print("="*70)
    print(f"Defect Recall:         {results['summary']['system_recall']*100:.2f}% (Target: >= 99%)")
    print(f"Catastrophic Escapes:  {results['summary']['false_negatives_escapes']}")
    print(f"Cost-Weighted Score:   {results['summary']['cost_score_50FN_1FP']} (FN=50x FP penalty)")
    print(f"Drift MAE (168h):      {results['summary']['drift_metrics']['mae_original_scale_uA']:.4f} µA")
    print(f"Drift MAE (Log Scale): {results['summary']['drift_metrics']['mae_log_scale']:.4f}")
    print(f"Burn-In Hours Saved:   {results['summary']['drift_metrics']['chamber_hours_saved']} hours at 24h")
    print("="*70)
    print("\nBaseline Comparison Table:")
    print(results['baseline_df'].to_string(index=False))
    print("\nAblation Study Table:")
    print(ablation_df.to_string(index=False))
