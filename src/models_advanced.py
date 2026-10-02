"""
models_advanced.py - Advanced ML/NN Models for Module A and Module B
SIH26170: Space-Grade Semiconductor Latent Defect Screening

Phase 2-4 implementation:
- Certified recall threshold (Clopper-Pearson bound)
- Stacked meta-learner with isotonic calibration
- Heteroscedastic MLP deep ensemble
- Gaussian Process on log-drift
- Conformalised Quantile Regression (CQR) for calibrated 10/90 bands
- Autoencoder anomaly detector with per-feature explanation
- GMM density-based anomaly score
- Lot-level maverick detector
- Physics-informed head: v(t) = v0·(1 + α·(t/168)^γ)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from scipy.stats import beta as beta_dist
from sklearn.calibration import IsotonicRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import warnings


# =========================================================================
# Phase 2: Certified Recall (Conformal Risk Control)
# =========================================================================

def recall_lower_bound(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided Clopper-Pearson lower bound on recall."""
    if k == 0:
        return 0.0
    return float(beta_dist.ppf(1 - conf, k, n - k + 1))


def pick_certified_threshold(scores_defect: np.ndarray, target: float = 0.99,
                              conf: float = 0.95) -> Dict[str, Any]:
    """
    Choose the highest score threshold whose Clopper-Pearson lower bound
    on recall reaches the target.
    
    Returns dict with threshold, certified recall bound, and diagnostics.
    """
    s = np.sort(np.asarray(scores_defect))[::-1]  # High to low
    n = len(s)
    
    best_threshold = None
    best_bound = 0.0
    best_k = 0
    
    for t in s:
        k = int((s >= t).sum())
        bound = recall_lower_bound(k, n, conf)
        if bound >= target:
            if best_threshold is None or t > best_threshold:
                best_threshold = float(t)
                best_bound = bound
                best_k = k
    
    # Even if we can't certify at target, report best achievable
    max_certifiable = recall_lower_bound(n, n, conf) if n > 0 else 0.0
    
    return {
        'threshold': best_threshold,
        'certified_recall_bound': round(best_bound, 6) if best_threshold is not None else None,
        'n_calibration_defects': n,
        'defects_at_threshold': best_k,
        'target_recall': target,
        'confidence': conf,
        'max_certifiable_recall': round(max_certifiable, 6),
        'sufficient_defects': best_threshold is not None,
        'note': (f"With {n} calibration defects, max certifiable recall "
                 f"at {conf:.0%} confidence is {max_certifiable:.4f}")
    }


# =========================================================================
# Phase 2: Stacked Meta-Learner with Isotonic Calibration
# =========================================================================

class CalibratedStackedClassifier:
    """
    Replaces hand-tuned 0.45/0.55 fusion weights with a learned stacked
    meta-learner that outputs calibrated probabilities via isotonic regression.
    """
    
    def __init__(self):
        self.weights = None
        self.isotonic = None
        self.fitted = False
    
    def fit(self, score_matrix: np.ndarray, y_true: np.ndarray):
        """
        score_matrix: (n_samples, n_models) — raw anomaly scores from each detector
        y_true: binary labels
        """
        from sklearn.linear_model import LogisticRegression
        
        # Learn fusion weights via logistic regression
        lr = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs')
        lr.fit(score_matrix, y_true)
        self.weights = lr.coef_[0]
        self._lr = lr
        
        # Raw stacked scores
        raw_scores = lr.predict_proba(score_matrix)[:, 1]
        
        # Isotonic calibration
        self.isotonic = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        self.isotonic.fit(raw_scores, y_true)
        
        self.fitted = True
        return self
    
    def predict_proba(self, score_matrix: np.ndarray) -> np.ndarray:
        """Returns calibrated P(defect) for each sample."""
        if not self.fitted:
            raise ValueError("Must fit before predict_proba")
        raw_scores = self._lr.predict_proba(score_matrix)[:, 1]
        return self.isotonic.predict(raw_scores)


# =========================================================================
# Phase 3: Heteroscedastic MLP Deep Ensemble (PyTorch-free NumPy version)
# =========================================================================

class HeteroscedasticEnsemble:
    """
    Deep ensemble of heteroscedastic regressors.
    Uses sklearn's MLPRegressor as base (avoids PyTorch dependency).
    Each member predicts mean; ensemble variance from member disagreement.
    
    For full heteroscedastic NLL, install torch and use the HeteroMLP in Appendix A.2.
    """
    
    def __init__(self, n_members: int = 5, hidden_size: int = 64):
        self.n_members = n_members
        self.hidden_size = hidden_size
        self.members = []
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        from sklearn.neural_network import MLPRegressor
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        self.members = []
        for seed in range(self.n_members):
            mlp = MLPRegressor(
                hidden_layer_sizes=(self.hidden_size, self.hidden_size),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=seed * 7 + 42,
                early_stopping=True,
                validation_fraction=0.15,
                learning_rate_init=0.001,
            )
            mlp.fit(X_scaled, y)
            self.members.append(mlp)
        
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Returns mean, epistemic variance, and per-member predictions."""
        X_scaled = self.scaler.transform(X)
        preds = np.array([m.predict(X_scaled) for m in self.members])  # (n_members, n_samples)
        
        mu = preds.mean(axis=0)
        var_epistemic = preds.var(axis=0)  # Disagreement = epistemic uncertainty
        std = np.sqrt(var_epistemic)
        
        return {
            'mean': mu,
            'std': std,
            'lower_10': mu - 1.28 * std,  # ~10th percentile
            'upper_90': mu + 1.28 * std,   # ~90th percentile
            'per_member': preds,
        }


# =========================================================================
# Phase 3: Conformalised Quantile Regression
# =========================================================================

def cqr_calibrate(y_cal: np.ndarray, q_lo_cal: np.ndarray,
                   q_hi_cal: np.ndarray, alpha: float = 0.2) -> float:
    """
    Compute CQR conformity margin on calibration set.
    alpha = 0.2 → target 80% coverage of the [10, 90] band.
    """
    s = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(s)[min(k, n) - 1])


def cqr_predict_interval(q_lo: np.ndarray, q_hi: np.ndarray,
                          margin: float) -> Tuple[np.ndarray, np.ndarray]:
    """Calibrated interval: [q_lo - margin, q_hi + margin]."""
    return q_lo - margin, q_hi + margin


# =========================================================================
# Phase 3: Gaussian Process on Log-Drift
# =========================================================================

class GPDriftPredictor:
    """
    Gaussian Process regression on log-drift with lot-aware features.
    Well-calibrated uncertainty on small data; good independent check on NNs.
    """
    
    def __init__(self, n_restarts: int = 3, max_train: int = 2000):
        self.n_restarts = n_restarts
        self.max_train = max_train
        self.gp = None
        self.scaler = StandardScaler()
        self.fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        # Subsample if too large (GP scales O(n³))
        if len(X) > self.max_train:
            idx = np.random.default_rng(42).choice(len(X), self.max_train, replace=False)
            X = X[idx]
            y = y[idx]
        
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(X.shape[1])) + WhiteKernel(0.01)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            random_state=42,
            normalize_y=True,
        )
        self.gp.fit(X_scaled, y)
        self.fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        X_scaled = self.scaler.transform(X)
        mu, std = self.gp.predict(X_scaled, return_std=True)
        return {
            'mean': mu,
            'std': std,
            'lower_10': mu - 1.28 * std,
            'upper_90': mu + 1.28 * std,
        }


# =========================================================================
# Phase 4: Autoencoder Anomaly Detector (NumPy / sklearn fallback)
# =========================================================================

class AutoencoderDetector:
    """
    Autoencoder anomaly detector with per-feature reconstruction error.
    Uses sklearn MLPRegressor as identity-reconstruction model (avoids torch dependency).
    Per-feature squared error provides transparent attribution.
    """
    
    def __init__(self, bottleneck: int = 3, hidden: int = 16):
        self.bottleneck = bottleneck
        self.hidden = hidden
        self.model = None
        self.scaler = StandardScaler()
        self.threshold_95 = None
        self.fitted = False
    
    def fit(self, X_healthy: np.ndarray):
        from sklearn.neural_network import MLPRegressor
        
        self.scaler.fit(X_healthy)
        X_scaled = self.scaler.transform(X_healthy)
        
        # Train autoencoder as identity-reconstruction MLP
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden, self.bottleneck, self.hidden),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        )
        self.model.fit(X_scaled, X_scaled)
        
        # Compute threshold from training reconstruction errors
        recon = self.model.predict(X_scaled)
        errors = np.sum((recon - X_scaled) ** 2, axis=1)
        self.threshold_95 = float(np.percentile(errors, 95))
        self.threshold_99 = float(np.percentile(errors, 99))
        
        self.fitted = True
        return self
    
    def score(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Returns anomaly scores and per-feature reconstruction errors.
        """
        X_scaled = self.scaler.transform(X)
        recon = self.model.predict(X_scaled)
        per_feature_error = (recon - X_scaled) ** 2  # shape [n, d_in]
        total_score = per_feature_error.sum(axis=1)   # anomaly score
        
        # Normalise to [0, 1] using training percentiles
        score_normalised = np.clip(total_score / max(self.threshold_99, 1e-6), 0, 1)
        
        return {
            'anomaly_score': score_normalised,
            'reconstruction_error': total_score,
            'per_feature_error': per_feature_error,
            'threshold_95': self.threshold_95,
            'threshold_99': self.threshold_99,
        }


# =========================================================================
# Phase 4: GMM Density-Based Anomaly
# =========================================================================

class GMMDetector:
    """
    Gaussian Mixture Model on healthy features.
    Low log-likelihood → anomaly.
    """
    
    def __init__(self, n_components: int = 3):
        self.n_components = n_components
        self.gmm = None
        self.scaler = StandardScaler()
        self.threshold_5pct = None
        self.fitted = False
    
    def fit(self, X_healthy: np.ndarray):
        self.scaler.fit(X_healthy)
        X_scaled = self.scaler.transform(X_healthy)
        
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type='full',
            max_iter=200,
            random_state=42,
        )
        self.gmm.fit(X_scaled)
        
        # Threshold: 5th percentile of healthy log-likelihoods
        scores = self.gmm.score_samples(X_scaled)
        self.threshold_5pct = float(np.percentile(scores, 5))
        
        self.fitted = True
        return self
    
    def score(self, X: np.ndarray) -> np.ndarray:
        """Returns anomaly scores: high = anomalous."""
        X_scaled = self.scaler.transform(X)
        ll = self.gmm.score_samples(X_scaled)
        # Invert: lower log-likelihood → higher anomaly score
        score = 1.0 / (1.0 + np.exp((ll - self.threshold_5pct) * 2.0))
        return np.clip(score, 0, 1)


# =========================================================================
# Phase 4: Lot-Level Maverick Detector
# =========================================================================

def detect_maverick_lots(df: pd.DataFrame, historical_stats: Optional[Dict] = None,
                         z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Flags the whole lot when its distribution departs from historical lots.
    Returns dataframe with lot-level health status.
    """
    lot_stats = []
    
    for lot_id, grp in df.groupby('lot_id'):
        lot_median_v24 = grp['value_24h'].median()
        lot_mad_v24 = 1.4826 * np.median(np.abs(grp['value_24h'] - lot_median_v24))
        lot_mad_v24 = max(lot_mad_v24, 1e-4)
        n_parts = len(grp)
        
        # Fraction of parts with Z > 2 within the lot
        if 'z_pat_24h' in grp.columns:
            frac_elevated = (grp['z_pat_24h'].abs() > 2.0).mean()
        else:
            frac_elevated = 0.0
        
        lot_stats.append({
            'lot_id': lot_id,
            'n_parts': n_parts,
            'lot_median_v24': round(lot_median_v24, 4),
            'lot_mad_v24': round(lot_mad_v24, 4),
            'frac_elevated': round(frac_elevated, 4),
        })
    
    lot_df = pd.DataFrame(lot_stats)
    
    # Compare each lot's median to the global distribution of lot medians
    global_lot_median = lot_df['lot_median_v24'].median()
    global_lot_mad = 1.4826 * np.median(np.abs(lot_df['lot_median_v24'] - global_lot_median))
    global_lot_mad = max(global_lot_mad, 1e-4)
    
    lot_df['z_lot_vs_global'] = (lot_df['lot_median_v24'] - global_lot_median) / global_lot_mad
    lot_df['is_maverick'] = (lot_df['z_lot_vs_global'].abs() > z_threshold) | (lot_df['frac_elevated'] > 0.30)
    lot_df['lot_health'] = lot_df['is_maverick'].map({True: 'MAVERICK', False: 'NOMINAL'})
    
    return lot_df


# =========================================================================
# Phase 5: Exact Mahalanobis Decomposition
# =========================================================================

def mahalanobis_decomposition(x: np.ndarray, mu: np.ndarray,
                               cov_inv: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    """
    Exact Mahalanobis decomposition per feature.
    d² = Σᵢ diffᵢ · (Σ⁻¹ diff)ᵢ, so contributions sum exactly to the distance.
    """
    diff = x - mu
    contrib = diff * (cov_inv @ diff)
    d_squared = float(contrib.sum())
    
    feature_contributions = []
    for i, name in enumerate(feature_names):
        feature_contributions.append({
            'feature': name,
            'deviation': round(float(diff[i]), 4),
            'contribution': round(float(contrib[i]), 4),
            'fraction': round(float(contrib[i]) / max(d_squared, 1e-8), 4),
        })
    
    # Sort by contribution (descending)
    feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
    
    return {
        'mahalanobis_distance_squared': round(d_squared, 4),
        'mahalanobis_distance': round(float(np.sqrt(max(d_squared, 0))), 4),
        'feature_contributions': feature_contributions,
        'top_contributors': [fc['feature'] for fc in feature_contributions[:3]],
    }


# =========================================================================
# Phase 5: Counterfactual Margin
# =========================================================================

def compute_counterfactual(part_row: pd.Series, model_fn, feature_cols: List[str],
                            target_col: str = 'value_24h') -> Dict[str, Any]:
    """
    "This part would have passed if its 24h reading were at most X µA."
    Line search for the ensemble; closed form for rule thresholds.
    """
    current_val = float(part_row[target_col])
    lot_med = float(part_row.get('lot_median_v24', current_val))
    
    # Binary search for threshold where decision flips to ACCEPT
    lo, hi = lot_med * 0.5, current_val
    best_pass_val = None
    
    for _ in range(30):  # Binary search iterations
        mid = (lo + hi) / 2.0
        test_row = part_row.copy()
        test_row[target_col] = mid
        
        try:
            score = model_fn(test_row)
            if score < 0.5:  # Would pass
                best_pass_val = mid
                lo = mid  # Try higher
            else:
                hi = mid  # Try lower
        except Exception:
            hi = mid
    
    if best_pass_val is not None:
        return {
            'counterfactual_threshold': round(best_pass_val, 2),
            'current_value': round(current_val, 2),
            'margin': round(current_val - best_pass_val, 2),
            'narrative': f"This part would have passed if its 24h reading were at most {best_pass_val:.2f} µA (measured {current_val:.2f} µA)."
        }
    else:
        return {
            'counterfactual_threshold': None,
            'current_value': round(current_val, 2),
            'margin': None,
            'narrative': f"No feasible counterfactual found — defect characteristics extend beyond single-feature correction."
        }


# =========================================================================
# Phase 5: Nearest Healthy Neighbours
# =========================================================================

def find_nearest_healthy(part_features: np.ndarray, healthy_features: np.ndarray,
                          healthy_ids: List[str], k: int = 3) -> List[Dict[str, Any]]:
    """
    The k most similar passing parts, using Euclidean distance in normalised feature space.
    """
    dists = np.sqrt(np.sum((healthy_features - part_features) ** 2, axis=1))
    nearest_idx = np.argsort(dists)[:k]
    
    neighbours = []
    for idx in nearest_idx:
        neighbours.append({
            'part_id': healthy_ids[idx],
            'distance': round(float(dists[idx]), 4),
        })
    
    return neighbours


# =========================================================================
# Phase 6: Mission-Life Safety Slope (Arrhenius)
# =========================================================================

def arrhenius_safety_slope(datasheet_limit: float, v0: float,
                            mission_life_hours: float = 87600,  # 10 years
                            Ea: float = 0.7,  # eV (illustrative)
                            T_stress_C: float = 125.0,
                            T_use_C: float = 55.0,
                            burnin_hours: float = 168.0) -> Dict[str, Any]:
    """
    Replaces arbitrary k·σ safety slope with physics-based Arrhenius derivation.
    
    AF = exp[(Ea/k)(1/T_use - 1/T_stress)]
    Equivalent field time = burnin_hours × AF
    Safety slope = max allowed drift such that field extrapolation stays below limit.
    """
    k_boltzmann = 8.617e-5  # eV/K
    T_stress_K = T_stress_C + 273.15
    T_use_K = T_use_C + 273.15
    
    AF = np.exp((Ea / k_boltzmann) * (1.0 / T_use_K - 1.0 / T_stress_K))
    equiv_field_hours = burnin_hours * AF
    equiv_field_years = equiv_field_hours / 8760.0
    
    # Maximum allowable drift over burn-in period
    max_drift_burnin = datasheet_limit - v0  # µA headroom
    
    # Safety slope: must not exceed limit at mission end
    # Field drift extrapolated = (slope * burnin_hours) * AF * (mission_life / equiv_field_hours)
    safety_slope = max_drift_burnin / (burnin_hours * (mission_life_hours / equiv_field_hours))
    safety_slope = max(safety_slope, 1e-6)
    
    return {
        'acceleration_factor': round(float(AF), 1),
        'equiv_field_hours': round(float(equiv_field_hours), 0),
        'equiv_field_years': round(float(equiv_field_years), 2),
        'max_drift_burnin_uA': round(float(max_drift_burnin), 2),
        'safety_slope_uA_per_hr': round(float(safety_slope), 6),
        'Ea_eV': Ea,
        'T_stress_C': T_stress_C,
        'T_use_C': T_use_C,
        'mission_life_hours': mission_life_hours,
    }


# =========================================================================
# Phase 6: Pseudo-Labels for Label-Free Mode
# =========================================================================

def generate_pseudo_labels(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.Series:
    """
    Generates pseudo-labels from the full training trajectory.
    Uses robust Z of 168h log-drift as the labelling criterion.
    Same pipeline then runs in label-free mode.
    """
    if 'value_168h' not in df.columns or df['value_168h'].isna().all():
        warnings.warn("Cannot generate pseudo-labels without value_168h")
        return pd.Series(0, index=df.index)
    
    log_drift = np.log(np.maximum(df['value_168h'], 1e-5)) - np.log(np.maximum(df['value_0h'], 1e-5))
    median_drift = np.median(log_drift)
    mad_drift = 1.4826 * np.median(np.abs(log_drift - median_drift))
    mad_drift = max(mad_drift, 1e-6)
    
    z_scores = (log_drift - median_drift) / mad_drift
    pseudo_labels = (z_scores > z_threshold).astype(int)
    
    return pseudo_labels
