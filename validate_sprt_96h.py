import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.generate_physics import generate_physics_dataset, split_lots
from src.preprocess import BurnInPreprocessor
from src.module_a import DynamicOutlierDetector
from src.module_b import DriftPredictor
from src.decision import ScreeningDecisionEngine, SequentialScreeningEngine

def main():
    print("Generating data...")
    df = generate_physics_dataset()
    train_df, _, test_df = split_lots(df)
    
    print("Training pipeline...")
    prep = BurnInPreprocessor()
    prep.fit(train_df)
    train_p = prep.transform(train_df)
    test_p = prep.transform(test_df)
    
    mod_a = DynamicOutlierDetector()
    mod_a.fit(train_p)
    mod_b = DriftPredictor()
    mod_b.fit(train_p)
    
    print("Evaluating test set...")
    pred_a = mod_a.predict_detailed(test_p)
    pred_b = mod_b.predict(test_p)
    
    engine = ScreeningDecisionEngine()
    final_df = engine.evaluate(pred_a, pred_b)
    
    sprt = SequentialScreeningEngine()
    print("\n--- SPRT 24h Decision ---")
    df_24h = sprt.decide_at_24h(final_df)
    print(df_24h['sprt_decision_24h'].value_counts())
    
    print("\n--- SPRT 96h Re-Decision ---")
    df_96h = sprt.decide_at_96h(df_24h, test_p['value_96h'].values)
    
    if 'sprt_decision_96h' in df_96h.columns:
        print("Final 96h Decisions for Uncertain Parts:")
        # Only look at parts that were UNCERTAIN at 24h
        uncertain_at_24h = df_96h[df_96h['sprt_decision_24h'] == 'UNCERTAIN']
        print(uncertain_at_24h['sprt_decision_96h'].value_counts())
        
        # Cross tabulate with actual defect status
        import pandas as pd
        ct = pd.crosstab(uncertain_at_24h['is_defect'], uncertain_at_24h['sprt_decision_96h'])
        print("\nDefect Catch Rate at 96h:")
        print(ct)
    else:
        print("96h decision logic not fully implemented or returned.")

if __name__ == "__main__":
    main()
