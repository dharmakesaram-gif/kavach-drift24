"""
models_physics.py - Physics-Informed Head & Teacher-Student Models
SIH26170: Space-Grade Semiconductor Latent Defect Screening

Phase 3-6 implementation:
- Physics-informed head: v(t) = v0 · (1 + α · (t/168)^γ)
  Network outputs α and γ (γ constrained positive). Extrapolates sensibly.
  γ < 1: sub-linear healthy ageing; γ > 1: super-linear TDDB-like defect.
- Teacher-student with privileged information:
  Teacher trained on all 4 readings (0h, 24h, 96h, 168h).
  Student trained only on 0h/24h to match teacher's soft targets.
  At inference only 0h/24h are used.
"""

import numpy as np
import warnings
from typing import Dict, Optional, List
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# =========================================================================
# Physics-Informed Drift Head
# =========================================================================

class PhysicsInformedDriftHead:
    """
    Physics-informed 168h drift predictor.

    Model: v(t) = v0 · (1 + α · (t/168)^γ)

    The model predicts α (total fractional drift) and γ (time exponent):
    - γ < 1: sub-linear, healthy NBTI-like ageing (saturating drift)
    - γ ≈ 1: linear drift (electromigration-like)
    - γ > 1: super-linear, accelerating TDDB-like defect

    With only 0h and 24h readings, a single (α, γ) pair is identified.
    We use an informative prior on γ learned from healthy lots to constrain
    the solution space.
    """

    def __init__(self, gamma_prior_mean: float = 0.5, gamma_prior_std: float = 0.3):
        self.gamma_prior_mean = gamma_prior_mean
        self.gamma_prior_std = gamma_prior_std
        self.alpha_model = None
        self.gamma_model = None
        self.scaler = StandardScaler()
        self.healthy_gamma_stats = None
        self.fitted = False

    def fit(self, X: np.ndarray, v0: np.ndarray, v168: np.ndarray,
            v24: Optional[np.ndarray] = None, v96: Optional[np.ndarray] = None,
            is_healthy: Optional[np.ndarray] = None):
        """
        Fits the physics head by:
        1. Computing ground-truth α from (v0, v168).
        2. Estimating γ from multi-point trajectory if 24h/96h available.
        3. Learning γ prior from healthy parts.
        4. Training regressors for α and γ from input features.
        """
        n = len(X)

        # Compute ground-truth α: v168 = v0 · (1 + α), so α = (v168/v0) - 1
        alpha_true = (v168 / np.maximum(v0, 1e-5)) - 1.0
        alpha_true = np.maximum(alpha_true, 1e-6)

        # Estimate γ from multi-point trajectory
        # v(24) = v0·(1 + α·(24/168)^γ) → (v24/v0 - 1)/α = (24/168)^γ
        # → γ = log((v24/v0 - 1)/α) / log(24/168)
        if v24 is not None:
            drift_frac_24 = np.maximum((v24 / np.maximum(v0, 1e-5)) - 1.0, 1e-8)
            ratio_24 = np.clip(drift_frac_24 / np.maximum(alpha_true, 1e-8), 1e-6, 1.0 - 1e-6)
            gamma_est = np.log(ratio_24) / np.log(24.0 / 168.0)
            gamma_est = np.clip(gamma_est, 0.05, 3.0)
        else:
            # Default to prior
            gamma_est = np.full(n, self.gamma_prior_mean)

        # Refine γ with 96h if available
        if v96 is not None:
            drift_frac_96 = np.maximum((v96 / np.maximum(v0, 1e-5)) - 1.0, 1e-8)
            ratio_96 = np.clip(drift_frac_96 / np.maximum(alpha_true, 1e-8), 1e-6, 1.0 - 1e-6)
            gamma_96 = np.log(ratio_96) / np.log(96.0 / 168.0)
            gamma_96 = np.clip(gamma_96, 0.05, 3.0)
            # Average 24h and 96h estimates
            valid_96 = np.isfinite(gamma_96)
            gamma_est[valid_96] = 0.5 * gamma_est[valid_96] + 0.5 * gamma_96[valid_96]

        # Learn healthy γ prior
        if is_healthy is not None:
            healthy_gammas = gamma_est[is_healthy.astype(bool)]
            if len(healthy_gammas) > 5:
                self.healthy_gamma_stats = {
                    'mean': float(np.median(healthy_gammas)),
                    'std': float(np.std(healthy_gammas)),
                }
                self.gamma_prior_mean = self.healthy_gamma_stats['mean']
                self.gamma_prior_std = max(self.healthy_gamma_stats['std'], 0.05)

        # Scale features
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Train regressors
        self.alpha_model = MLPRegressor(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
            max_iter=500, random_state=42, early_stopping=True,
            validation_fraction=0.15, learning_rate_init=0.001,
        )
        self.alpha_model.fit(X_scaled, np.log(np.maximum(alpha_true, 1e-8)))

        self.gamma_model = MLPRegressor(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
            max_iter=500, random_state=43, early_stopping=True,
            validation_fraction=0.15, learning_rate_init=0.001,
        )
        self.gamma_model.fit(X_scaled, gamma_est)

        self.fitted = True
        return self

    def predict(self, X: np.ndarray, v0: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predicts 168h value using physics model.

        Returns:
            mean: predicted v(168)
            alpha: predicted fractional drift
            gamma: predicted time exponent
            v_at_96h: predicted v(96)
            is_accelerating: gamma > 1.0 (super-linear drift flag)
        """
        if not self.fitted:
            raise ValueError("PhysicsInformedDriftHead must be fitted first.")

        X_scaled = self.scaler.transform(X)

        # Predict α (in log space for positivity)
        log_alpha = self.alpha_model.predict(X_scaled)
        alpha = np.exp(np.clip(log_alpha, -10, 5))

        # Predict γ (constrained positive via softplus)
        gamma_raw = self.gamma_model.predict(X_scaled)
        gamma = np.clip(gamma_raw, 0.05, 3.0)  # Constrained positive

        # Physics model: v(168) = v0 · (1 + α)
        pred_168 = v0 * (1.0 + alpha)
        pred_96 = v0 * (1.0 + alpha * (96.0 / 168.0) ** gamma)

        return {
            'mean': pred_168,
            'alpha': alpha,
            'gamma': gamma,
            'v_at_96h': pred_96,
            'is_accelerating': gamma > 1.0,
            'gamma_prior_mean': self.gamma_prior_mean,
        }


# =========================================================================
# Teacher-Student with Privileged Information (PDF §6.2)
# =========================================================================

class TeacherStudentDriftModel:
    """
    Teacher-student distillation for 168h drift forecasting.

    Teacher: Trained on ALL four readings (0h, 24h, 96h, 168h) to learn
    the true latent defect signature with full trajectory information.

    Student: Trained ONLY on 0h/24h features to match:
    1. Teacher's 168h regression targets (soft targets)
    2. Teacher's anomaly labels (soft classification)

    At inference, only the student is used — the problem-statement
    constraint (only 0h and 24h available) is preserved.
    """

    def __init__(self, n_teacher_members: int = 3, n_student_members: int = 5,
                 hidden: int = 64, temperature: float = 2.0):
        self.n_teacher_members = n_teacher_members
        self.n_student_members = n_student_members
        self.hidden = hidden
        self.temperature = temperature

        self.teacher_members = []
        self.student_members = []
        self.teacher_scaler = StandardScaler()
        self.student_scaler = StandardScaler()
        self.fitted = False

    def _get_teacher_features(self, X_base: np.ndarray, v0: np.ndarray,
                              v24: np.ndarray, v96: np.ndarray,
                              v168: np.ndarray) -> np.ndarray:
        """Builds teacher feature matrix with all four readings."""
        log_v0 = np.log(np.maximum(v0, 1e-5))
        log_v24 = np.log(np.maximum(v24, 1e-5))
        log_v96 = np.log(np.maximum(v96, 1e-5))
        log_v168 = np.log(np.maximum(v168, 1e-5))

        # Privileged features: full trajectory shape
        delta_24 = log_v24 - log_v0
        delta_96 = log_v96 - log_v0
        delta_168 = log_v168 - log_v0
        accel_24_96 = (log_v96 - log_v24) / 72.0 - (log_v24 - log_v0) / 24.0
        accel_96_168 = (log_v168 - log_v96) / 72.0 - (log_v96 - log_v24) / 72.0

        return np.column_stack([
            X_base, log_v96, log_v168,
            delta_24, delta_96, delta_168,
            accel_24_96, accel_96_168
        ])

    def fit(self, X_student: np.ndarray, v0: np.ndarray, v24: np.ndarray,
            v96: np.ndarray, v168: np.ndarray,
            y_delta: np.ndarray, y_labels: Optional[np.ndarray] = None):
        """
        1. Train teacher on full trajectory features → soft regression targets.
        2. Train student on 0h/24h features to match teacher's outputs.
        """
        # --- Teacher training ---
        X_teacher = self._get_teacher_features(X_student, v0, v24, v96, v168)
        self.teacher_scaler.fit(X_teacher)
        X_t_scaled = self.teacher_scaler.transform(X_teacher)

        self.teacher_members = []
        teacher_preds = []
        for seed in range(self.n_teacher_members):
            mlp = MLPRegressor(
                hidden_layer_sizes=(self.hidden, self.hidden),
                activation='relu', solver='adam', max_iter=500,
                random_state=seed * 11 + 100, early_stopping=True,
                validation_fraction=0.15,
            )
            mlp.fit(X_t_scaled, y_delta)
            self.teacher_members.append(mlp)
            teacher_preds.append(mlp.predict(X_t_scaled))

        # Teacher ensemble soft targets
        soft_targets = np.mean(teacher_preds, axis=0)

        # --- Student training (matches teacher's soft targets) ---
        self.student_scaler.fit(X_student)
        X_s_scaled = self.student_scaler.transform(X_student)

        # Blend hard targets (ground truth) with soft targets (teacher) via temperature
        # Higher temperature = more reliance on teacher's smoothed output
        blend_weight = 1.0 / (1.0 + self.temperature)
        blended_targets = blend_weight * y_delta + (1.0 - blend_weight) * soft_targets

        self.student_members = []
        for seed in range(self.n_student_members):
            mlp = MLPRegressor(
                hidden_layer_sizes=(self.hidden, self.hidden),
                activation='relu', solver='adam', max_iter=500,
                random_state=seed * 7 + 42, early_stopping=True,
                validation_fraction=0.15,
            )
            mlp.fit(X_s_scaled, blended_targets)
            self.student_members.append(mlp)

        self.fitted = True
        return self

    def predict(self, X_student: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Student-only inference using 0h/24h features.
        Returns ensemble mean, std, and per-member predictions.
        """
        if not self.fitted:
            raise ValueError("TeacherStudentDriftModel must be fitted first.")

        X_s_scaled = self.student_scaler.transform(X_student)
        preds = np.array([m.predict(X_s_scaled) for m in self.student_members])

        mu = preds.mean(axis=0)
        std = preds.std(axis=0)

        return {
            'mean': mu,
            'std': std,
            'lower_10': mu - 1.28 * std,
            'upper_90': mu + 1.28 * std,
            'per_member': preds,
        }

    def predict_teacher(self, X_student: np.ndarray, v0: np.ndarray,
                        v24: np.ndarray, v96: np.ndarray,
                        v168: np.ndarray) -> Dict[str, np.ndarray]:
        """Teacher prediction (uses all 4 readings — for ablation/comparison only)."""
        X_teacher = self._get_teacher_features(X_student, v0, v24, v96, v168)
        X_t_scaled = self.teacher_scaler.transform(X_teacher)
        preds = np.array([m.predict(X_t_scaled) for m in self.teacher_members])

        mu = preds.mean(axis=0)
        std = preds.std(axis=0)

        return {
            'mean': mu,
            'std': std,
            'lower_10': mu - 1.28 * std,
            'upper_90': mu + 1.28 * std,
        }


# =========================================================================
# Stacked Module B Ensemble (GBM + DE + GP + Physics + Teacher-Student)
# =========================================================================

class StackedDriftEnsemble:
    """
    Stacks all Module B forecasters with learned weights optimised on
    validation lots by minimising MAE.

    Members:
        - GBM (existing DriftPredictor)
        - Deep Ensemble (HeteroscedasticEnsemble)
        - GP (GPDriftPredictor)
        - Physics Head (PhysicsInformedDriftHead)
        - Teacher-Student (TeacherStudentDriftModel)

    Weights learned via Ridge regression on validation set predictions.
    """

    def __init__(self):
        self.weights = None
        self.bias = 0.0
        self.member_names: List[str] = []
        self.fitted = False

    def fit(self, member_preds: Dict[str, np.ndarray], y_true: np.ndarray):
        """
        Learns stacking weights from member predictions.
        member_preds: {'gbm': array, 'deep_ensemble': array, ...}
        y_true: ground truth log-delta values.
        """
        from sklearn.linear_model import Ridge

        self.member_names = list(member_preds.keys())
        X_stack = np.column_stack([member_preds[k] for k in self.member_names])

        reg = Ridge(alpha=1.0)
        reg.fit(X_stack, y_true)
        self.weights = reg.coef_
        self.bias = reg.intercept_
        self.fitted = True
        return self

    def predict(self, member_preds: Dict[str, np.ndarray]) -> np.ndarray:
        """Returns stacked prediction."""
        if not self.fitted:
            raise ValueError("StackedDriftEnsemble must be fitted first.")

        X_stack = np.column_stack([member_preds[k] for k in self.member_names])
        return X_stack @ self.weights + self.bias

    def get_member_weights(self) -> Dict[str, float]:
        """Returns learned weight for each ensemble member."""
        return dict(zip(self.member_names, self.weights.tolist()))
