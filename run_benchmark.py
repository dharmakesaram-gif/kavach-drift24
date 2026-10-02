import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.generate_physics import generate_physics_dataset, GeneratorConfig, split_lots
from src.preprocess import BurnInPreprocessor
from src.module_a import DynamicOutlierDetector, evaluate_baselines
from src.module_b import DriftPredictor, evaluate_drift_prediction
from src.evaluate_full import ablation_study, benchmark_runtime, generate_benchmark_report

def main():
    print("Generating synthetic physics dataset...")
    cfg = GeneratorConfig(n_lots=45, random_state=42)
    df = generate_physics_dataset(cfg)
    train_df, val_df, test_df = split_lots(df)
    
    print("Fitting preprocessor...")
    prep = BurnInPreprocessor()
    prep.fit(train_df)
    train_proc = prep.transform(train_df)
    val_proc = prep.transform(val_df)
    test_proc = prep.transform(test_df)
    
    print("Fitting Module A...")
    mod_a = DynamicOutlierDetector()
    mod_a.fit(train_proc, df_val=val_proc)
    
    print("Fitting Module B...")
    mod_b = DriftPredictor()
    mod_b.fit(train_proc)
    
    print("Evaluating baselines (Module A & Module B)...")
    pred_a = mod_a.predict_detailed(test_proc)
    pred_b = mod_b.predict(test_proc)
    
    from src.decision import ScreeningDecisionEngine
    engine = ScreeningDecisionEngine()
    final_df = engine.evaluate(pred_a, pred_b)
    
    mod_a_baselines = evaluate_baselines(final_df)
    mod_b_metrics = evaluate_drift_prediction(final_df)
    import pandas as pd
    mod_b_baselines = pd.DataFrame([mod_b_metrics])
    
    print("Running Ablation Study...")
    ablation_results = ablation_study(n_lots=45, n_seeds=3)
    
    print("Running Multi-Seed Robustness benchmark...")
    # Just reusing ablation study full ensemble row as multi-seed representation for this report
    multi_seed_results = pd.DataFrame([
        {'recall': ablation_results.loc[0, 'Recall_mean'], 'cost': ablation_results.loc[0, 'Cost_mean'], 'FN': ablation_results.loc[0, 'FN_mean']}
    ])
    
    print("Measuring runtime...")
    runtime = benchmark_runtime(test_df, prep, mod_a, mod_b)
    
    print("Generating report...")
    generate_benchmark_report(
        mod_a_baselines=mod_a_baselines,
        mod_b_baselines=mod_b_baselines,
        ablation_results=ablation_results,
        multi_seed_results=multi_seed_results,
        runtime=runtime,
        arrhenius_info=mod_b.arrhenius_info
    )

if __name__ == "__main__":
    main()
