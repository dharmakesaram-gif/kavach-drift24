"""
module_a.py - Multi-Layered Dynamic Outlier Detection System
SIH26170: Space-Grade Semiconductor Latent Defect Screening

Implements:
1. Rule Layer: Dynamic Part Average Testing (AEC-Q001 DPAT, Robust Z > 3.5, excessive drift)
2. Statistical Layer: Robust Mahalanobis Distance (Minimum Covariance Determinant)
3. ML Layer: Isolation Forest on lot-normalized parametric deviations
4. Advanced Detectors: Autoencoder (per-feature error), GMM (density), LOF (local density)
5. Calibrated Fusion: Stacked meta-learner with isotonic calibration (replaces hand-tuned weights)
6. Maverick Lot Pre-Screen: Flags contaminated lots before individual part assessment
7. Certified Recall: Clopper-Pearson bound on conformal threshold
8. Three-Tier Output: ACCEPT, REVIEW (borderline safeguard), REJECT
"""

import numpy as np
import pandas as pd
import warnings
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import IsolationForest
from sklearn.covariance import MinCovDet, EmpiricalCovariance
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import recall_score, precision_score, f1_score, confusion_matrix

from src.models_advanced import (
    AutoencoderDetector, GMMDetector, CalibratedStackedClassifier,
    detect_maverick_lots, pick_certified_threshold, recall_lower_bound,
)


class DynamicOutlierDetector:
    """
    Module A: Dynamic Outlier Detection Engine.
    Detects latent defects at 24h that escape static limits by benchmarking against lot distributions.

    Ensemble members:
        1. Rule engine (DPAT Z-scores, drift, ratio, low-side)
        2. MCD Mahalanobis distance (robust covariance)
        3. Isolation Forest
        4. Autoencoder reconstruction error (per-feature attribution)
        5. GMM density-based anomaly score
        6. Local Outlier Factor (non-parametric local density)

    Fusion:
        CalibratedStackedClassifier (logistic + isotonic) when labels available,
        otherwise equal-weight average with sigmoid calibration.
    """

    def __init__(
        self,
        z_rule_threshold: float = 3.5,
        drift_rule_threshold: float = 2.2,
        ratio_rule_threshold: float = 2.8,
        cost_fn_weight: float = 50.0,
        cost_fp_weight: float = 1.0,
        random_state: int = 42
    ):
        self.z_rule_threshold = z_rule_threshold
        self.drift_rule_threshold = drift_rule_threshold
        self.ratio_rule_threshold = ratio_rule_threshold
        self.cost_fn_weight = cost_fn_weight
        self.cost_fp_weight = cost_fp_weight
        self.random_state = random_state

        self.feature_cols = [
            'z_pat_0h',
            'z_pat_24h',
            'z_drift_24h',
            'ratio_to_lot_24h',
            'excess_slope_24h'
        ]

        # Core detectors (Phase 0)
        self.mcd = None
        self.iforest = None
        self.mcd_mean = None
        self.mcd_cov_inv = None

        # Advanced detectors (Phase 4)
        self.autoencoder = None
        self.gmm_detector = None
        self.lof_scores_train = None  # LOF is transductive, stored for reference

        # Calibrated fusion (Phase 2)
        self.stacked_classifier = None
        self.use_calibrated_fusion = False

        # Certified threshold (Phase 2)
        self.certified_threshold = None
        self.certified_recall_bound = None

        # Maverick lot info
        self.maverick_lots = None

        # Calibrated decision thresholds
        self.reject_threshold = 0.65
        self.review_threshold = 0.40
        self.fitted = False

    def fit(self, df_train: pd.DataFrame, df_val: Optional[pd.DataFrame] = None) -> 'DynamicOutlierDetector':
        """
        Fits all detector layers on training data.

        1. Maverick lot pre-screen
        2. MCD Mahalanobis on healthy/all features
        3. Isolation Forest on healthy/all features
        4. Autoencoder on healthy features (per-feature reconstruction error)
        5. GMM on healthy features (density-based anomaly)
        6. If labels available: CalibratedStackedClassifier for fusion
        7. If validation set: calibrate thresholds and certify recall
        """
        train_features = df_train[self.feature_cols].copy()
        has_labels = 'is_defect' in df_train.columns and df_train['is_defect'].sum() > 0

        # Filter healthy parts if labels available
        if has_labels:
            nominal_data = train_features[df_train['is_defect'] == 0].values
        else:
            nominal_data = train_features.values

        # --- Step 0: Maverick Lot Pre-Screen ---
        self.maverick_lots = detect_maverick_lots(df_train)

        # --- Step 1: Robust Mahalanobis (MCD) ---
        try:
            self.mcd = MinCovDet(random_state=self.random_state, support_fraction=0.85)
            self.mcd.fit(nominal_data)
            self.mcd_mean = self.mcd.location_
            cov = self.mcd.covariance_ + np.eye(len(self.feature_cols)) * 1e-4
            self.mcd_cov_inv = np.linalg.pinv(cov)
        except Exception:
            self.mcd = EmpiricalCovariance()
            self.mcd.fit(nominal_data)
            self.mcd_mean = self.mcd.location_
            cov = self.mcd.covariance_ + np.eye(len(self.feature_cols)) * 1e-3
            self.mcd_cov_inv = np.linalg.pinv(cov)

        # --- Step 2: Isolation Forest ---
        self.iforest = IsolationForest(
            n_estimators=150, contamination=0.03, max_samples='auto',
            random_state=self.random_state, n_jobs=-1
        )
        self.iforest.fit(nominal_data)

        # --- Step 3: Autoencoder ---
        try:
            self.autoencoder = AutoencoderDetector(bottleneck=3, hidden=16)
            self.autoencoder.fit(nominal_data)
        except Exception as e:
            warnings.warn(f"Autoencoder fit failed: {e}. Skipping AE layer.")
            self.autoencoder = None

        # --- Step 4: GMM ---
        try:
            self.gmm_detector = GMMDetector(n_components=3)
            self.gmm_detector.fit(nominal_data)
        except Exception as e:
            warnings.warn(f"GMM fit failed: {e}. Skipping GMM layer.")
            self.gmm_detector = None

        # --- Step 5: Calibrated Stacked Fusion (if labels) ---
        if has_labels:
            self._fit_calibrated_fusion(df_train)

        self.fitted = True

        # --- Step 6: Threshold Calibration + Certified Recall ---
        if df_val is not None and 'is_defect' in df_val.columns:
            self.calibrate_thresholds(df_val)
            self._certify_recall(df_val)

        return self

    def _fit_calibrated_fusion(self, df_train: pd.DataFrame):
        """Fits CalibratedStackedClassifier on the score matrix from all detectors."""
        X = df_train[self.feature_cols].values
        y = df_train['is_defect'].values

        # Compute per-detector scores
        score_matrix = self._build_score_matrix(X)

        try:
            self.stacked_classifier = CalibratedStackedClassifier()
            self.stacked_classifier.fit(score_matrix, y)
            self.use_calibrated_fusion = True
        except Exception as e:
            warnings.warn(f"Calibrated fusion fit failed: {e}. Using default weights.")
            self.use_calibrated_fusion = False

    def _build_score_matrix(self, X: np.ndarray) -> np.ndarray:
        """Builds the (n_samples, n_detectors) score matrix for fusion."""
        scores = []

        # 1. Mahalanobis
        scores.append(self._calc_mahalanobis_scores(X))

        # 2. Isolation Forest
        scores.append(self._calc_iforest_scores(X))

        # 3. Autoencoder
        if self.autoencoder is not None and self.autoencoder.fitted:
            ae_result = self.autoencoder.score(X)
            scores.append(ae_result['anomaly_score'])
        else:
            scores.append(np.zeros(len(X)))

        # 4. GMM
        if self.gmm_detector is not None and self.gmm_detector.fitted:
            scores.append(self.gmm_detector.score(X))
        else:
            scores.append(np.zeros(len(X)))

        return np.column_stack(scores)

    def _calc_mahalanobis_scores(self, X: np.ndarray) -> np.ndarray:
        """Computes regularized Mahalanobis distances and normalizes to [0, 1]."""
        diff = X - self.mcd_mean
        dists = np.sqrt(np.sum(np.dot(diff, self.mcd_cov_inv) * diff, axis=1))
        # Normalization using robust scaling (99th percentile of chi-squared df=5 is ~15.1)
        scores = 1.0 / (1.0 + np.exp(-(dists - 5.0) / 2.0))
        return np.clip(scores, 0.0, 1.0)

    def _calc_iforest_scores(self, X: np.ndarray) -> np.ndarray:
        """Inverts IsolationForest decision function so high score = high anomaly."""
        raw_scores = self.iforest.score_samples(X)  # Typical range [-0.8, -0.3]
        # Map: -0.65 -> 0.9, -0.4 -> 0.1
        scores = 1.0 / (1.0 + np.exp((raw_scores + 0.50) * 15.0))
        return np.clip(scores, 0.0, 1.0)

    def _calc_lof_scores(self, X: np.ndarray) -> np.ndarray:
        """Runs LOF (transductive — fits and predicts on the same batch)."""
        lof = LocalOutlierFactor(n_neighbors=min(20, max(5, len(X) // 10)),
                                 contamination=0.03, novelty=False)
        lof_pred = lof.fit_predict(X)
        neg_scores = lof.negative_outlier_factor_
        # Normalise: typical range is around -1.0 to -10.0
        # More negative → more anomalous
        score = 1.0 / (1.0 + np.exp((neg_scores + 1.3) * 5.0))
        return np.clip(score, 0.0, 1.0)

    def predict_detailed(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Executes all detector layers, calculates individual and ensemble scores,
        determines 3-tier action (ACCEPT, REVIEW, REJECT), and compiles reason codes.

        Includes:
        - Maverick lot flagging
        - Rule engine, MCD, IForest, AE, GMM, LOF scores
        - Calibrated fusion or default weighted average
        - Certified threshold (if calibrated)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before predict_detailed() can be called.")

        X = df[self.feature_cols].values
        n_samples = len(df)

        # --- Maverick Lot Flagging ---
        maverick_lot_ids = set()
        if self.maverick_lots is not None:
            mav_df = self.maverick_lots[self.maverick_lots['is_maverick']]
            maverick_lot_ids = set(mav_df['lot_id'].values)

        # Also detect maverick lots in the prediction batch itself
        batch_mav = detect_maverick_lots(df)
        batch_mav_ids = set(batch_mav[batch_mav['is_maverick']]['lot_id'].values)
        maverick_lot_ids = maverick_lot_ids | batch_mav_ids

        # --- Layer 1: Rule Engine ---
        rule_flags = np.zeros(n_samples, dtype=bool)
        rule_reasons = [[] for _ in range(n_samples)]

        z0 = df['z_pat_0h'].values
        z24 = df['z_pat_24h'].values
        z_drift = df['z_drift_24h'].values
        ratio24 = df['ratio_to_lot_24h'].values

        for i in range(n_samples):
            # High-side rules
            if z0[i] > self.z_rule_threshold:
                rule_flags[i] = True
                rule_reasons[i].append(f"DPAT_0H_EXCEEDED (Z={z0[i]:.1f}σ > {self.z_rule_threshold}σ)")
            if z24[i] > self.z_rule_threshold:
                rule_flags[i] = True
                rule_reasons[i].append(f"DPAT_24H_EXCEEDED (Z={z24[i]:.1f}σ > {self.z_rule_threshold}σ)")
            if z_drift[i] > self.drift_rule_threshold:
                rule_flags[i] = True
                rule_reasons[i].append(f"EXCESSIVE_DRIFT_VELOCITY (Drift Z={z_drift[i]:.1f}σ > {self.drift_rule_threshold}σ)")
            if ratio24[i] > self.ratio_rule_threshold:
                rule_flags[i] = True
                rule_reasons[i].append(f"LOT_RATIO_ANOMALOUS ({ratio24[i]:.1f}x lot median)")
            # Low-side rules — abnormally low leakage is also a screening signal
            if z0[i] < -self.z_rule_threshold:
                rule_flags[i] = True
                rule_reasons[i].append(f"DPAT_0H_LOW_ANOMALY (Z={z0[i]:.1f}σ < -{self.z_rule_threshold}σ)")
            if z24[i] < -self.z_rule_threshold:
                rule_flags[i] = True
                rule_reasons[i].append(f"DPAT_24H_LOW_ANOMALY (Z={z24[i]:.1f}σ < -{self.z_rule_threshold}σ)")

            # Maverick lot flag
            lot_id = df.iloc[i].get('lot_id', '')
            if lot_id in maverick_lot_ids:
                rule_reasons[i].append(f"MAVERICK_LOT_FLAGGED (lot {lot_id})")

        # --- Layer 2: Statistical Mahalanobis ---
        mahal_scores = self._calc_mahalanobis_scores(X)

        # --- Layer 3: Isolation Forest ---
        iforest_scores = self._calc_iforest_scores(X)

        # --- Layer 4: Autoencoder ---
        ae_scores = np.zeros(n_samples)
        ae_per_feature = None
        if self.autoencoder is not None and self.autoencoder.fitted:
            ae_result = self.autoencoder.score(X)
            ae_scores = ae_result['anomaly_score']
            ae_per_feature = ae_result['per_feature_error']

        # --- Layer 5: GMM ---
        gmm_scores = np.zeros(n_samples)
        if self.gmm_detector is not None and self.gmm_detector.fitted:
            gmm_scores = self.gmm_detector.score(X)

        # --- Layer 6: LOF (transductive on batch) ---
        lof_scores = np.zeros(n_samples)
        if n_samples >= 10:  # LOF needs minimum neighbours
            try:
                lof_scores = self._calc_lof_scores(X)
            except Exception:
                pass  # LOF can fail on very small or degenerate batches

        # --- Ensemble Fusion ---
        if self.use_calibrated_fusion and self.stacked_classifier is not None:
            # Calibrated stacked fusion using learned weights + isotonic calibration
            score_matrix = self._build_score_matrix(X)
            ensemble_score = self.stacked_classifier.predict_proba(score_matrix)
        else:
            # Default: weighted average (MCD + IForest dominate, others contribute)
            ensemble_score = (
                0.25 * mahal_scores +
                0.25 * iforest_scores +
                0.15 * ae_scores +
                0.15 * gmm_scores +
                0.10 * lof_scores +
                0.10 * np.maximum(np.abs(z24) / 10.0, 0)  # Rule-score contribution
            )
            ensemble_score = np.clip(ensemble_score, 0.0, 1.0)

        # Rule triggers boost ensemble score
        ensemble_score = np.where(rule_flags, np.maximum(ensemble_score, 0.88), ensemble_score)

        # --- Three-Tier Classification ---
        # Use certified threshold if available, otherwise calibrated/default thresholds
        rej_t = self.certified_threshold if self.certified_threshold is not None else self.reject_threshold
        rev_t = self.review_threshold

        decisions = []
        for i in range(n_samples):
            sc = ensemble_score[i]
            rf = rule_flags[i]

            if rf or sc >= rej_t:
                decisions.append("REJECT")
                if not rule_reasons[i]:
                    rule_reasons[i].append(f"ML_ENSEMBLE_CONFIDENCE ({sc:.2f} >= {rej_t:.2f})")
            elif sc >= rev_t:
                decisions.append("REVIEW")
                rule_reasons[i].append(f"BORDERLINE_ANOMALY (Score {sc:.2f} in review band [{rev_t:.2f}, {rej_t:.2f}))")
            else:
                decisions.append("ACCEPT")

        result = df.copy()
        result['score_rule_flag'] = rule_flags.astype(int)
        result['score_mahalanobis'] = np.round(mahal_scores, 4)
        result['score_iforest'] = np.round(iforest_scores, 4)
        result['score_autoencoder'] = np.round(ae_scores, 4)
        result['score_gmm'] = np.round(gmm_scores, 4)
        result['score_lof'] = np.round(lof_scores, 4)
        result['anomaly_score_mod_a'] = np.round(ensemble_score, 4)
        result['decision_mod_a'] = decisions
        result['reason_codes_mod_a'] = ["; ".join(r) if r else "NOMINAL_LOT_DISTRIBUTION" for r in rule_reasons]
        result['is_maverick_lot'] = df['lot_id'].isin(maverick_lot_ids).astype(int)

        return result

    def _certify_recall(self, df_val: pd.DataFrame):
        """
        Uses Clopper-Pearson bound to find the highest threshold that certifies
        >= 99% recall at 95% confidence on the validation defects.
        """
        res = self.predict_detailed(df_val)
        y_true = df_val['is_defect'].values
        scores_defect = res.loc[y_true == 1, 'anomaly_score_mod_a'].values

        cert = pick_certified_threshold(scores_defect, target=0.99, conf=0.95)
        if cert['sufficient_defects'] and cert['threshold'] is not None:
            self.certified_threshold = cert['threshold']
            self.certified_recall_bound = cert['certified_recall_bound']
        else:
            # Not enough defects to certify — keep calibrated threshold
            self.certified_threshold = None
            self.certified_recall_bound = cert.get('max_certifiable_recall')
            warnings.warn(cert['note'])

    def calibrate_thresholds(self, df_val: pd.DataFrame, target_recall: float = 0.99) -> Dict[str, float]:
        """
        Grid searches review & reject thresholds on validation data to guarantee target_recall (>=99%)
        while minimizing cost-weighted penalty (Cost = 50 * FN + 1 * FP).
        Warns if target recall is unreachable; calibrates both thresholds independently.
        """
        y_true = df_val['is_defect'].values
        res = self.predict_detailed(df_val)
        scores = res['anomaly_score_mod_a'].values

        best_cost = float('inf')
        best_reject = 0.65
        best_review = 0.40
        found_feasible = False

        # Fine grid for review threshold
        review_thresholds = np.linspace(0.10, 0.85, 50)
        # Independent grid for reject threshold offset
        reject_offsets = np.linspace(0.10, 0.40, 15)

        for review_t in review_thresholds:
            y_pred_positive = (scores >= review_t) | (res['score_rule_flag'] == 1)
            rec = recall_score(y_true, y_pred_positive, zero_division=0)

            if rec >= target_recall:
                cm = confusion_matrix(y_true, y_pred_positive, labels=[0, 1])
                tn, fp, fn, tp = cm.ravel()
                cost = (self.cost_fn_weight * fn) + (self.cost_fp_weight * fp)

                if cost < best_cost:
                    best_cost = cost
                    best_review = review_t
                    found_feasible = True
                    # Find optimal reject threshold independently
                    best_r_cost = float('inf')
                    for offset in reject_offsets:
                        r_thresh = min(0.95, review_t + offset)
                        y_reject = (scores >= r_thresh) | (res['score_rule_flag'] == 1)
                        cm_r = confusion_matrix(y_true, y_reject, labels=[0, 1])
                        _, fp_r, fn_r, _ = cm_r.ravel()
                        r_cost = (self.cost_fn_weight * fn_r) + (self.cost_fp_weight * fp_r)
                        if r_cost < best_r_cost:
                            best_r_cost = r_cost
                            best_reject = r_thresh

        if not found_feasible:
            warnings.warn(
                f"Target recall {target_recall:.2%} is unreachable on validation data. "
                f"Using lowest available threshold. Consider generating more calibration defects.",
                UserWarning
            )
            # Use lowest threshold that maximises recall
            best_review = float(review_thresholds[0])
            best_reject = best_review + 0.20

        self.review_threshold = round(best_review, 3)
        self.reject_threshold = round(best_reject, 3)

        return {
            'target_recall': target_recall,
            'achieved_feasible': found_feasible,
            'review_threshold': self.review_threshold,
            'reject_threshold': self.reject_threshold,
            'best_cost': best_cost
        }

    def get_detector_report(self) -> Dict[str, Any]:
        """Returns a summary of all fitted detector layers and their status."""
        return {
            'mcd_fitted': self.mcd is not None,
            'iforest_fitted': self.iforest is not None,
            'autoencoder_fitted': self.autoencoder is not None and self.autoencoder.fitted,
            'gmm_fitted': self.gmm_detector is not None and self.gmm_detector.fitted,
            'calibrated_fusion': self.use_calibrated_fusion,
            'certified_threshold': self.certified_threshold,
            'certified_recall_bound': self.certified_recall_bound,
            'reject_threshold': self.reject_threshold,
            'review_threshold': self.review_threshold,
            'n_maverick_lots': len(self.maverick_lots[self.maverick_lots['is_maverick']]) if self.maverick_lots is not None else 0,
        }


def evaluate_baselines(df_results: pd.DataFrame) -> pd.DataFrame:
    """
    Compares Module A against:
    1. Static Datasheet Limit Screening
    2. Global Z-score (ignoring lot differences)
    3. Module A Dynamic Lot-Aware Ensemble
    """
    y_true = df_results['is_defect'].values

    # 1. Baseline: Static Limit
    # Static fails part if any observed reading exceeds datasheet limit
    static_flag = (df_results['value_0h'] > df_results['datasheet_limit']) | \
                  (df_results['value_24h'] > df_results['datasheet_limit'])
    rec_static = recall_score(y_true, static_flag, zero_division=0)
    prec_static = precision_score(y_true, static_flag, zero_division=0)
    cm_static = confusion_matrix(y_true, static_flag)
    fn_static = cm_static[1, 0] if cm_static.shape == (2, 2) else 0
    fp_static = cm_static[0, 1] if cm_static.shape == (2, 2) else 0
    cost_static = 50 * fn_static + 1 * fp_static

    # 2. Baseline: Global Z-Score (ignoring lots)
    glob_med = df_results['value_24h'].median()
    glob_mad = 1.4826 * np.median(np.abs(df_results['value_24h'] - glob_med))
    glob_z = (df_results['value_24h'] - glob_med) / max(glob_mad, 1e-4)
    glob_flag = glob_z > 3.0
    rec_glob = recall_score(y_true, glob_flag, zero_division=0)
    prec_glob = precision_score(y_true, glob_flag, zero_division=0)
    cm_glob = confusion_matrix(y_true, glob_flag)
    fn_glob = cm_glob[1, 0] if cm_glob.shape == (2, 2) else 0
    fp_glob = cm_glob[0, 1] if cm_glob.shape == (2, 2) else 0
    cost_glob = 50 * fn_glob + 1 * fp_glob

    # 3. Module A Ensemble: Flagged if REVIEW or REJECT
    mod_a_flag = df_results['decision_mod_a'].isin(['REVIEW', 'REJECT'])
    rec_mod_a = recall_score(y_true, mod_a_flag, zero_division=0)
    prec_mod_a = precision_score(y_true, mod_a_flag, zero_division=0)
    cm_mod_a = confusion_matrix(y_true, mod_a_flag)
    fn_mod_a = cm_mod_a[1, 0] if cm_mod_a.shape == (2, 2) else 0
    fp_mod_a = cm_mod_a[0, 1] if cm_mod_a.shape == (2, 2) else 0
    cost_mod_a = 50 * fn_mod_a + 1 * fp_mod_a

    metrics = [
        {
            'Method': 'Static Datasheet Limit Only',
            'Recall (Catch Rate)': f"{rec_static*100:.1f}%",
            'Precision': f"{prec_static*100:.1f}%",
            'False Negatives (Catastrophic)': fn_static,
            'False Positives (Scrap)': fp_static,
            'Cost Score (50*FN + 1*FP)': cost_static
        },
        {
            'Method': 'Global Z-Score (No Lot Awareness)',
            'Recall (Catch Rate)': f"{rec_glob*100:.1f}%",
            'Precision': f"{prec_glob*100:.1f}%",
            'False Negatives (Catastrophic)': fn_glob,
            'False Positives (Scrap)': fp_glob,
            'Cost Score (50*FN + 1*FP)': cost_glob
        },
        {
            'Method': 'Module A: Dynamic Lot-Aware Ensemble',
            'Recall (Catch Rate)': f"{rec_mod_a*100:.1f}%",
            'Precision': f"{prec_mod_a*100:.1f}%",
            'False Negatives (Catastrophic)': fn_mod_a,
            'False Positives (Scrap)': fp_mod_a,
            'Cost Score (50*FN + 1*FP)': cost_mod_a
        }
    ]
    return pd.DataFrame(metrics)
