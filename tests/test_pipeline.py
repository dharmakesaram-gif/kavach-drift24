"""
test_pipeline.py - Automated Unit & Integration Tests for SIH26170
Tests data generation, preprocessing, outlier detection, drift prediction, and decision fusion.
"""

import unittest
import numpy as np
import pandas as pd

from src.generate import generate_burnin_dataset, split_lots
from src.preprocess import BurnInPreprocessor
from src.module_a import DynamicOutlierDetector
from src.module_b import DriftPredictor
from src.decision import ScreeningDecisionEngine
from src.explain import generate_qa_report_card, compute_local_feature_contributions


class TestBurnInPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Generates a small test dataset for rapid testing."""
        cls.df = generate_burnin_dataset(n_lots=6, parts_per_lot_range=(50, 80), random_state=123)
        cls.train_df, cls.val_df, cls.test_df = split_lots(cls.df, train_ratio=0.6, val_ratio=0.2, random_state=123)

    def test_01_data_generator_schema(self):
        """Verifies synthetic dataset columns, types, and physical sanity bounds."""
        expected_cols = {'part_id', 'lot_id', 'value_0h', 'value_24h', 'value_96h', 'value_168h', 'is_defect', 'defect_type', 'datasheet_limit'}
        self.assertTrue(expected_cols.issubset(set(self.df.columns)))
        self.assertGreater(len(self.df), 200)
        # All physical leakage currents must be strictly positive
        self.assertTrue((self.df['value_0h'] > 0).all())
        self.assertTrue((self.df['value_168h'] > 0).all())
        # Static pass must not flag subtle in-spec defects
        subtle = self.df[self.df['defect_type'] == 'subtle_in_spec']
        if len(subtle) > 0:
            self.assertTrue((subtle['value_0h'] <= subtle['datasheet_limit']).all())

    def test_02_preprocessor_robust_dpat(self):
        """Verifies robust log-transforms, DPAT z-score calculation, and small lot shrinkage."""
        pre = BurnInPreprocessor(small_lot_threshold=30, shrinkage_prior_weight=15.0)
        pre.fit(self.train_df)
        trans = pre.transform(self.test_df)
        
        self.assertIn('z_pat_0h', trans.columns)
        self.assertIn('z_pat_24h', trans.columns)
        self.assertIn('z_drift_24h', trans.columns)
        self.assertFalse(trans['z_pat_24h'].isna().any())

    def test_03_module_a_outlier_detection(self):
        """Verifies Module A multi-layer execution, scores, and 3-tier classification."""
        pre = BurnInPreprocessor().fit(self.train_df)
        train_p = pre.transform(self.train_df)
        test_p = pre.transform(self.test_df)
        
        mod_a = DynamicOutlierDetector(cost_fn_weight=50.0, cost_fp_weight=1.0)
        mod_a.fit(train_p)
        res_a = mod_a.predict_detailed(test_p)
        
        self.assertIn('anomaly_score_mod_a', res_a.columns)
        self.assertIn('decision_mod_a', res_a.columns)
        valid_tiers = {'ACCEPT', 'REVIEW', 'REJECT'}
        self.assertTrue(set(res_a['decision_mod_a'].unique()).issubset(valid_tiers))

    def test_04_module_b_drift_prediction(self):
        """Verifies Module B 168h forecast, quantile monotonicity, and safety slope calculation."""
        pre = BurnInPreprocessor().fit(self.train_df)
        train_p = pre.transform(self.train_df)
        test_p = pre.transform(self.test_df)
        
        mod_b = DriftPredictor(safety_k_sigma=3.0)
        mod_b.fit(train_p)
        res_b = mod_b.predict(test_p)
        
        self.assertIn('pred_v168', res_b.columns)
        self.assertIn('pred_v168_lower', res_b.columns)
        self.assertIn('pred_v168_upper', res_b.columns)
        # Quantile monotonicity check: lower <= median <= upper
        self.assertTrue((res_b['pred_v168_lower'] <= res_b['pred_v168'] + 1e-4).all())
        self.assertTrue((res_b['pred_v168'] <= res_b['pred_v168_upper'] + 1e-4).all())
        self.assertGreater(mod_b.safety_slope_limit, 0.0)

    def test_05_decision_engine_and_explainability(self):
        """Verifies multi-module decision matrix and QA report generation."""
        pre = BurnInPreprocessor().fit(self.train_df)
        test_p = pre.transform(self.test_df)
        
        mod_a = DynamicOutlierDetector().fit(test_p)
        mod_b = DriftPredictor().fit(test_p)
        
        res_a = mod_a.predict_detailed(test_p)
        res_b = mod_b.predict(test_p)
        
        engine = ScreeningDecisionEngine()
        final_df = engine.evaluate(res_a, res_b)
        
        self.assertIn('final_decision', final_df.columns)
        self.assertIn('burnin_hours_saved', final_df.columns)
        
        # Test QA report card generation on first row
        report = generate_qa_report_card(final_df.iloc[0])
        self.assertIn('narrative', report)
        self.assertIn('audit_trail', report)
        
        # Test feature contributions
        contribs = compute_local_feature_contributions(final_df.iloc[0])
        self.assertIsInstance(contribs, list)
        self.assertGreater(len(contribs), 0)


if __name__ == '__main__':
    unittest.main()
