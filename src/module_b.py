"""
module_b.py - Predictive Drift Forecaster & Early Rejection Engine
SIH26170: Space-Grade Semiconductor Latent Defect Screening

Implements:
1. Multi-Model Drift Forecaster:
   - Robust Linear Baseline (Huber Regressor)
   - Non-linear Gradient Boosting Regressor (XGBoost/LightGBM)
   - Quantile Confidence Bounds (10%, 50%, 90%)
2. Safety Slope Derivation:
   - Physics-based Arrhenius mission-life derivation (replaces arbitrary k·σ)
   - Acceleration factor AF from burn-in temperature to field temperature
   - Early rejection trigger when upper confidence bound exceeds safety margin
3. Validation Metrics:
   - Grouped Lot-CV MAE (original scale µA and log space)
   - Defect-restricted MAE (ensuring accuracy where it matters most)
   - Burn-in chamber hours saved estimator
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.models_advanced import arrhenius_safety_slope


class DriftPredictor:
    """
    Module B: Early Time-Series Drift Predictor.
    Takes 0h and 24h readings to forecast 168h parametric value and drift trajectory.
    """
    
    def __init__(
        self,
        safety_k_sigma: float = 3.0,
        quantile_alpha_low: float = 0.10,
        quantile_alpha_high: float = 0.90,
        random_state: int = 42,
        # Arrhenius physics parameters
        mission_life_hours: float = 87600,  # 10 years
        Ea: float = 0.7,  # eV (illustrative — JEDEC JEP122)
        T_stress_C: float = 125.0,
        T_use_C: float = 55.0,
    ):
        self.safety_k_sigma = safety_k_sigma
        self.quantile_alpha_low = quantile_alpha_low
        self.quantile_alpha_high = quantile_alpha_high
        self.random_state = random_state
        
        # Arrhenius physics parameters
        self.mission_life_hours = mission_life_hours
        self.Ea = Ea
        self.T_stress_C = T_stress_C
        self.T_use_C = T_use_C
        
        self.features_24h = [
            'log_v0',
            'log_v24',
            'delta_24h',
            'slope_24h',
            'rel_drift_24h',
            'z_pat_0h',
            'z_pat_24h',
            'z_drift_24h',
            'ratio_to_lot_v0',
            'ratio_to_lot_24h',
            'excess_slope_24h'
        ]
        
        # Models
        self.baseline_huber = None
        self.primary_regressor = None
        self.model_q10 = None
        self.model_q90 = None
        
        # Physics & Safety thresholds
        self.safety_slope_limit: float = 0.0
        self.healthy_mean_slope: float = 0.0
        self.healthy_std_slope: float = 0.0
        self.arrhenius_info: Dict[str, Any] = {}
        self.fitted = False

    def fit(self, df_train: pd.DataFrame) -> 'DriftPredictor':
        """
        Trains Huber baseline, gradient booster, and quantile regressors to predict 168h drift.
        Establishes safety slope limit from healthy population.
        """
        train_clean = df_train.dropna(subset=['value_168h'] + self.features_24h).copy()
        
        X = train_clean[self.features_24h].values
        # Predict delta_168 = log(v168) - log(v0)
        y_delta = (train_clean['log_v168'] - train_clean['log_v0']).values
        
        # 1. Huber Regressor Baseline
        self.baseline_huber = HuberRegressor(max_iter=2000, epsilon=1.35)
        self.baseline_huber.fit(X, y_delta)
        
        # 2. Primary Gradient Boosting Model (Median / Mean prediction)
        self.primary_regressor = GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.06,
            max_depth=4,
            loss='squared_error',
            random_state=self.random_state
        )
        self.primary_regressor.fit(X, y_delta)
        
        # 3. Quantile Regressors for Confidence Intervals
        self.model_q10 = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.07,
            max_depth=3,
            loss='quantile',
            alpha=self.quantile_alpha_low,
            random_state=self.random_state
        )
        self.model_q10.fit(X, y_delta)
        
        self.model_q90 = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.07,
            max_depth=3,
            loss='quantile',
            alpha=self.quantile_alpha_high,
            random_state=self.random_state
        )
        self.model_q90.fit(X, y_delta)
        
        # 4. Compute Safety Slope — Physics-based Arrhenius derivation
        if 'is_defect' in train_clean.columns:
            healthy_df = train_clean[train_clean['is_defect'] == 0]
        else:
            healthy_df = train_clean
            
        healthy_slopes = (healthy_df['value_168h'] - healthy_df['value_0h']) / 168.0  # µA/hr
        self.healthy_mean_slope = float(np.mean(healthy_slopes))
        self.healthy_std_slope = float(np.std(healthy_slopes))
        
        # Arrhenius physics-based safety slope (per-part, using median v0 and datasheet limit)
        median_v0 = float(healthy_df['value_0h'].median())
        datasheet_limit = float(healthy_df['datasheet_limit'].median()) if 'datasheet_limit' in healthy_df.columns else 50.0
        
        self.arrhenius_info = arrhenius_safety_slope(
            datasheet_limit=datasheet_limit,
            v0=median_v0,
            mission_life_hours=self.mission_life_hours,
            Ea=self.Ea,
            T_stress_C=self.T_stress_C,
            T_use_C=self.T_use_C,
        )
        physics_slope = self.arrhenius_info['safety_slope_uA_per_hr']
        
        # Legacy k·σ bound (kept as a floor)
        p99_5 = float(np.percentile(healthy_slopes, 99.5))
        p95_0 = float(np.percentile(healthy_slopes, 95.0))
        k_sigma_bound = self.healthy_mean_slope + self.safety_k_sigma * self.healthy_std_slope
        
        # Use the MORE CONSERVATIVE (smaller) of physics-based and statistical bounds
        self.safety_slope_limit = min(physics_slope, max(p99_5, k_sigma_bound))
        self.safety_slope_review = min(physics_slope * 0.7, max(p95_0, self.healthy_mean_slope + 1.8 * self.healthy_std_slope))
        
        self.fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Forecasts 168h values and confidence bounds, computes predicted drift rate,
        and applies early rejection safety checks at 24h.
        """
        if not self.fitted:
            raise ValueError("DriftPredictor must be fitted before predict() can be called.")
            
        out = df.copy()
        X = out[self.features_24h].values
        v0 = out['value_0h'].values
        v24 = out['value_24h'].values
        log_v0 = out['log_v0'].values
        
        # Forecast delta_168 = log(v168) - log(v0)
        pred_delta_huber = self.baseline_huber.predict(X)
        pred_delta_gbr = self.primary_regressor.predict(X)
        pred_delta_q10 = self.model_q10.predict(X)
        pred_delta_q90 = self.model_q90.predict(X)
        
        # Ensure quantile monotonic consistency: q10 <= median <= q90
        pred_delta_q10 = np.minimum(pred_delta_q10, pred_delta_gbr)
        pred_delta_q90 = np.maximum(pred_delta_q90, pred_delta_gbr)
        
        # Convert back from log-delta to original scale: v168 = exp(log_v0 + delta)
        pred_v168_huber = np.round(np.exp(log_v0 + pred_delta_huber), 4)
        pred_v168_median = np.round(np.exp(log_v0 + pred_delta_gbr), 4)
        pred_v168_lower = np.round(np.exp(log_v0 + pred_delta_q10), 4)
        pred_v168_upper = np.round(np.exp(log_v0 + pred_delta_q90), 4)
        
        # Compute predicted drift rate over 168h using upper bound (conservative for space)
        pred_slope_median = (pred_v168_median - v0) / 168.0
        pred_slope_upper = (pred_v168_upper - v0) / 168.0
        
        # Early rejection decisions at 24h based on drift forecast:
        obs_slope_24h = (v24 - v0) / 24.0
        flag_slope_reject = (pred_slope_upper > self.safety_slope_limit) | (pred_slope_median > self.safety_slope_limit)
        flag_slope_review = (pred_slope_upper > self.safety_slope_review) | (obs_slope_24h > self.safety_slope_review)
        
        limit = out['datasheet_limit'].values if 'datasheet_limit' in out.columns else np.full(len(out), 50.0)
        flag_limit_breach = pred_v168_upper >= limit * 0.95
        
        decisions = []
        reasons = []
        for i in range(len(out)):
            r = []
            if flag_limit_breach[i]:
                decisions.append("REJECT")
                r.append(f"PRED_168H_LIMIT_BREACH (Forecast {pred_v168_upper[i]:.1f}µA >= limit {limit[i]:.1f}µA)")
            elif flag_slope_reject[i]:
                decisions.append("REJECT")
                r.append(f"SAFETY_SLOPE_EXCEEDED (Rate {pred_slope_upper[i]:.4f} > Limit {self.safety_slope_limit:.4f} µA/h)")
            elif flag_slope_review[i]:
                decisions.append("REVIEW")
                r.append(f"SAFETY_SLOPE_ELEVATED (Drift rate {pred_slope_upper[i]:.4f} > Review Threshold {self.safety_slope_review:.4f} µA/h)")
            else:
                decisions.append("ACCEPT")
                r.append("NOMINAL_DRIFT_PROJECTION")
                
            reasons.append("; ".join(r))
            
        out['pred_v168_huber'] = pred_v168_huber
        out['pred_v168'] = pred_v168_median
        out['pred_v168_lower'] = pred_v168_lower
        out['pred_v168_upper'] = pred_v168_upper
        out['pred_drift_rate_168h'] = np.round(pred_slope_median, 5)
        out['pred_upper_drift_rate'] = np.round(pred_slope_upper, 5)
        out['safety_slope_threshold'] = round(self.safety_slope_limit, 5)
        out['safety_slope_review_threshold'] = round(self.safety_slope_review, 5)
        out['decision_mod_b'] = decisions
        out['reason_codes_mod_b'] = reasons
        
        return out


def evaluate_drift_prediction(df_predicted: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes Drift Prediction Accuracy metrics:
    - MAE on original physical scale (µA)
    - MAE in natural log space
    - Defect-restricted MAE (accuracy specifically on latent defects)
    - R-squared score
    - Burn-in chamber hours saved by early 24h rejection
    """
    if 'value_168h' not in df_predicted.columns:
        raise ValueError("Ground truth value_168h required for evaluation.")
        
    y_true = df_predicted['value_168h'].values
    y_pred = df_predicted['pred_v168'].values
    
    log_y_true = np.log(np.maximum(y_true, 1e-5))
    log_y_pred = np.log(np.maximum(y_pred, 1e-5))
    
    mae_orig = mean_absolute_error(y_true, y_pred)
    mae_log = mean_absolute_error(log_y_true, log_y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Baseline comparison (Huber)
    if 'pred_v168_huber' in df_predicted.columns:
        mae_huber = mean_absolute_error(y_true, df_predicted['pred_v168_huber'].values)
    else:
        mae_huber = None
        
    # Metrics restricted to defective parts
    metrics_defects = {}
    if 'is_defect' in df_predicted.columns and df_predicted['is_defect'].sum() > 0:
        def_mask = df_predicted['is_defect'] == 1
        mae_defect = mean_absolute_error(y_true[def_mask], y_pred[def_mask])
        metrics_defects['mae_defects_only'] = round(float(mae_defect), 4)
    else:
        metrics_defects['mae_defects_only'] = None
        
    # Economic impact: chamber hours saved
    # Standard burn-in is 168 hours. Rejecting a defective part at 24h saves 144 chamber hours per part!
    # BUG FIX: Only count hours saved for TRUE defects caught (true positives), not all rejects
    is_rejected = (df_predicted['decision_mod_b'] == 'REJECT')
    if 'is_defect' in df_predicted.columns:
        true_defect_rejects = (is_rejected & (df_predicted['is_defect'] == 1)).sum()
        false_positive_rejects = (is_rejected & (df_predicted['is_defect'] == 0)).sum()
    else:
        true_defect_rejects = int(is_rejected.sum())
        false_positive_rejects = 0
    chamber_hours_saved = int(true_defect_rejects) * (168 - 24)
    fp_chamber_cost = int(false_positive_rejects) * (168 - 24)  # wasted hours on false alarms
    
    return {
        'mae_original_scale_uA': round(float(mae_orig), 4),
        'mae_log_scale': round(float(mae_log), 4),
        'mae_huber_baseline_uA': round(float(mae_huber), 4) if mae_huber is not None else None,
        'r2_score': round(float(r2), 4),
        'early_rejections_at_24h': int(is_rejected.sum()),
        'true_defect_rejections': int(true_defect_rejects),
        'false_positive_rejections': int(false_positive_rejects),
        'chamber_hours_saved': int(chamber_hours_saved),
        'fp_chamber_cost_hours': int(fp_chamber_cost),
        **metrics_defects
    }
