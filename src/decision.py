"""
decision.py - Unified Multi-Module Decision Engine
SIH26170: Space-Grade Semiconductor Latent Defect Screening

Fuses Module A (Dynamic Outlier Detection) and Module B (Time-Series Drift Predictor):
- Implements the strict space-grade screening truth matrix
- Assigns granular standardized Reason Codes (AEC-Q001 & MIL-STD-883)
- Evaluates Early Burn-In Termination at 24h
- Sequential 24h/96h SPRT decision (ACCEPT/REJECT/UNCERTAIN at 24h, re-decide at 96h)
- Produces final disposition: ACCEPT, REVIEW, REJECT, UNCERTAIN
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from scipy.stats import norm


class ScreeningDecisionEngine:
    """
    Fuses Module A outlier classifications with Module B drift projections.
    """

    def __init__(self, review_sensitivity: str = 'strict'):
        self.review_sensitivity = review_sensitivity

    def evaluate(self, df_mod_a: pd.DataFrame, df_mod_b: pd.DataFrame) -> pd.DataFrame:
        """
        Merges results from Module A and Module B to produce final screening decision.
        """
        merged = df_mod_a.copy()

        # Merge Module B prediction columns if not already present
        b_cols = [
            'pred_v168', 'pred_v168_lower', 'pred_v168_upper',
            'pred_drift_rate_168h', 'pred_upper_drift_rate',
            'safety_slope_threshold', 'decision_mod_b', 'reason_codes_mod_b'
        ]
        for col in b_cols:
            if col in df_mod_b.columns and col not in merged.columns:
                merged[col] = df_mod_b[col]

        final_decisions = []
        primary_reasons = []
        all_codes_list = []
        early_reject_flags = []
        hours_saved_list = []
        risk_scores = []

        for idx, row in merged.iterrows():
            dec_a = row.get('decision_mod_a', 'ACCEPT')
            dec_b = row.get('decision_mod_b', 'ACCEPT')
            score_a = row.get('anomaly_score_mod_a', 0.0)

            reasons_a = str(row.get('reason_codes_mod_a', '')).split('; ')
            reasons_b = str(row.get('reason_codes_mod_b', '')).split('; ')
            combined_reasons = [r for r in reasons_a + reasons_b if r and r != 'NOMINAL_LOT_DISTRIBUTION' and r != 'NOMINAL_DRIFT_PROJECTION']

            # Decision Matrix Logic
            # 1. Any Outlier detected by Module A is instant REJECT
            if dec_a == 'REJECT':
                final_dec = 'REJECT'
                early_reject = True
                hours_saved = 144  # 168h - 24h
                prim_reason = reasons_a[0] if reasons_a else "MODULE_A_DYNAMIC_PAT_OUTLIER"
                risk = max(score_a, 0.95)

            # 2. Module A Normal/Borderline, but Module B predicts upper bound exceeds safety slope or static limit
            elif dec_b == 'REJECT':
                final_dec = 'REJECT'
                early_reject = True
                hours_saved = 144
                prim_reason = reasons_b[0] if reasons_b else "MODULE_B_PREDICTED_SAFETY_BREACH"
                risk = min(max(score_a + 0.35, 0.85), 1.0)

            # 3. Borderline cases: Module A flagged REVIEW or Module B flagged REVIEW
            elif dec_a == 'REVIEW' or dec_b == 'REVIEW':
                if dec_a == 'REVIEW' and dec_b == 'REVIEW':
                    final_dec = 'REJECT'  # Compounded risk across both static PAT and trajectory drift!
                    early_reject = True
                    hours_saved = 144
                    prim_reason = "COMPOUND_BORDERLINE_ELEVATION (Flagged by both PAT and Drift models)"
                    risk = 0.80
                else:
                    final_dec = 'REVIEW'
                    early_reject = False
                    hours_saved = 0
                    prim_reason = combined_reasons[0] if combined_reasons else "INSPECTOR_BENCH_RETEST_REQUIRED"
                    risk = min(max(score_a, 0.55), 1.0)

            # 4. Both models nominal
            else:
                final_dec = 'ACCEPT'
                early_reject = False
                hours_saved = 0
                prim_reason = "NOMINAL_WITHIN_LOT_SPEC"
                risk = min(float(score_a), 1.0)

            final_decisions.append(final_dec)
            primary_reasons.append(prim_reason)
            all_codes_list.append("; ".join(combined_reasons) if combined_reasons else "PASS_NOMINAL")
            early_reject_flags.append(early_reject)
            hours_saved_list.append(hours_saved)
            risk_scores.append(round(float(np.clip(risk, 0.0, 1.0)), 4))

        merged['final_decision'] = final_decisions
        merged['early_rejection_24h'] = early_reject_flags
        merged['burnin_hours_saved'] = hours_saved_list
        merged['composite_risk_score'] = risk_scores
        merged['primary_reason'] = primary_reasons
        merged['detailed_reason_codes'] = all_codes_list

        return merged


# =========================================================================
# Phase 6: Sequential 24h/96h SPRT Decision Engine (PDF §6.3)
# =========================================================================

class SequentialScreeningEngine:
    """
    Sequential Probability Ratio Test (SPRT) decision engine.

    At 24h: outputs ACCEPT, REJECT, or UNCERTAIN (continue to 96h).
    At 96h: re-decides with updated posterior from the additional observation.

    This trades chamber hours against escape risk instead of only flagging at 24h,
    and demonstrates the economic reasoning judges value.

    The SPRT bounds are:
        - Upper bound A = (1 - β) / α   (reject defective)
        - Lower bound B = β / (1 - α)   (accept healthy)

    where α = target false-negative rate, β = target false-positive rate.
    When the likelihood ratio Λ falls between B and A → UNCERTAIN.
    """

    def __init__(
        self,
        alpha: float = 0.01,   # Target FN rate (miss rate)
        beta: float = 0.05,    # Target FP rate
        prior_defect_rate: float = 0.015,
    ):
        self.alpha = alpha
        self.beta = beta
        self.prior_defect_rate = prior_defect_rate

        # SPRT boundaries
        self.A = (1.0 - beta) / max(alpha, 1e-6)    # Reject boundary
        self.B = beta / (1.0 - max(alpha, 1e-6))     # Accept boundary

    def _log_likelihood_ratio(self, z_score: float, drift_z: float,
                               pred_risk: float) -> float:
        """
        Computes log-likelihood ratio for the SPRT.
        Under H1 (defective): elevated z-scores and drift expected.
        Under H0 (healthy): z-scores near 0, low drift.

        Uses a simplified Gaussian model:
            H0: z ~ N(0, 1), drift_z ~ N(0, 1)
            H1: z ~ N(μ_defect, σ_defect), drift_z ~ N(μ_drift_defect, σ_drift_defect)
        """
        # Parameters for defective distribution (learned from data or heuristic)
        mu_defect_z = 4.0     # Expected z-score for defective parts
        sigma_defect_z = 2.0
        mu_defect_drift = 3.0
        sigma_defect_drift = 2.0

        # Log-likelihood under H1
        ll_h1_z = norm.logpdf(z_score, mu_defect_z, sigma_defect_z)
        ll_h1_drift = norm.logpdf(drift_z, mu_defect_drift, sigma_defect_drift)

        # Log-likelihood under H0
        ll_h0_z = norm.logpdf(z_score, 0, 1)
        ll_h0_drift = norm.logpdf(drift_z, 0, 1)

        # Combined log-likelihood ratio
        llr = (ll_h1_z + ll_h1_drift) - (ll_h0_z + ll_h0_drift)

        # Also incorporate the ML model's risk score as additional evidence
        if pred_risk > 0.01 and pred_risk < 0.99:
            llr += np.log(pred_risk / max(1 - pred_risk, 1e-6))

        return float(llr)

    def decide_at_24h(self, df_results: pd.DataFrame) -> pd.DataFrame:
        """
        First-stage SPRT decision at 24h.
        Returns df with columns: sprt_decision_24h, sprt_log_lr, sprt_reason.
        """
        out = df_results.copy()

        decisions = []
        log_lrs = []
        reasons = []
        hours_saved = []

        log_A = np.log(self.A)
        log_B = np.log(self.B)

        for _, row in out.iterrows():
            z24 = float(row.get('z_pat_24h', 0))
            z_drift = float(row.get('z_drift_24h', 0))
            risk = float(row.get('composite_risk_score', row.get('anomaly_score_mod_a', 0)))
            existing_dec = row.get('final_decision', 'ACCEPT')

            llr = self._log_likelihood_ratio(z24, z_drift, risk)
            log_lrs.append(round(llr, 4))

            if existing_dec == 'REJECT' or llr >= log_A:
                decisions.append('REJECT')
                reasons.append(f"SPRT_REJECT_24H (LLR={llr:.2f} >= A={log_A:.2f})")
                hours_saved.append(144)
            elif existing_dec == 'ACCEPT' and llr <= log_B:
                decisions.append('ACCEPT')
                reasons.append(f"SPRT_ACCEPT_24H (LLR={llr:.2f} <= B={log_B:.2f})")
                hours_saved.append(0)
            else:
                decisions.append('UNCERTAIN')
                reasons.append(
                    f"SPRT_UNCERTAIN_24H (B={log_B:.2f} < LLR={llr:.2f} < A={log_A:.2f}); CONTINUE_TO_96H"
                )
                hours_saved.append(0)  # Must continue burn-in to 96h

        out['sprt_decision_24h'] = decisions
        out['sprt_log_lr'] = log_lrs
        out['sprt_reason'] = reasons
        out['sprt_hours_saved'] = hours_saved

        return out

    def decide_at_96h(self, df_24h: pd.DataFrame, v96_values: np.ndarray,
                      lot_median_v96: Optional[np.ndarray] = None,
                      lot_mad_v96: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Second-stage SPRT decision at 96h for UNCERTAIN parts.

        Computes updated posterior with the additional 96h observation.
        Parts already ACCEPT/REJECT at 24h retain their decision.
        """
        out = df_24h.copy()

        # Only re-evaluate UNCERTAIN parts
        uncertain_mask = out['sprt_decision_24h'] == 'UNCERTAIN'

        if not uncertain_mask.any():
            out['sprt_decision_96h'] = out['sprt_decision_24h']
            out['sprt_reason_96h'] = out['sprt_reason']
            return out

        final_decisions = out['sprt_decision_24h'].copy()
        final_reasons = out['sprt_reason'].copy()
        final_hours = out['sprt_hours_saved'].copy()

        v0 = out['value_0h'].values
        v24 = out['value_24h'].values

        # Compute 96h z-scores if lot stats provided
        if lot_median_v96 is not None and lot_mad_v96 is not None:
            z_96 = (v96_values - lot_median_v96) / np.maximum(lot_mad_v96, 1e-4)
        else:
            # Estimate from batch
            med_96 = np.median(v96_values[np.isfinite(v96_values)])
            mad_96 = 1.4826 * np.median(np.abs(v96_values[np.isfinite(v96_values)] - med_96))
            mad_96 = max(mad_96, 1e-4)
            z_96 = (v96_values - med_96) / mad_96

        log_A = np.log(self.A)
        log_B = np.log(self.B)

        for i in range(len(out)):
            if not uncertain_mask.iloc[i]:
                continue

            # Update LLR with 96h evidence
            prior_llr = float(out.iloc[i].get('sprt_log_lr', 0))
            z96 = float(z_96[i]) if np.isfinite(z_96[i]) else 0.0

            # Drift acceleration: is 24h→96h slope steeper than 0h→24h?
            slope_0_24 = (v24[i] - v0[i]) / 24.0
            slope_24_96 = (v96_values[i] - v24[i]) / 72.0
            accel = slope_24_96 / max(abs(slope_0_24), 1e-6)

            # Additional evidence from 96h
            additional_llr = self._log_likelihood_ratio(z96, accel, 0.5)
            updated_llr = prior_llr + 0.5 * additional_llr  # Weighted update

            if updated_llr >= log_A:
                final_decisions.iloc[i] = 'REJECT'
                final_reasons.iloc[i] = (
                    f"SPRT_REJECT_96H (Updated LLR={updated_llr:.2f}; "
                    f"Z_96h={z96:.1f}σ, Accel={accel:.2f}x)"
                )
                final_hours.iloc[i] = 72  # 168h - 96h saved
            elif updated_llr <= log_B:
                final_decisions.iloc[i] = 'ACCEPT'
                final_reasons.iloc[i] = (
                    f"SPRT_ACCEPT_96H (Updated LLR={updated_llr:.2f}; "
                    f"96h trajectory nominal)"
                )
                final_hours.iloc[i] = 0
            else:
                # Still uncertain — default to REVIEW (conservative for space grade)
                final_decisions.iloc[i] = 'REVIEW'
                final_reasons.iloc[i] = (
                    f"SPRT_REVIEW_96H (LLR={updated_llr:.2f} still inconclusive; "
                    f"Z_96h={z96:.1f}σ — recommend full 168h burn-in)"
                )
                final_hours.iloc[i] = 0

        out['sprt_decision_96h'] = final_decisions
        out['sprt_reason_96h'] = final_reasons
        out['sprt_hours_saved'] = final_hours

        return out

    def compute_economics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Computes economic trade-off report for the sequential decision.
        """
        dec_col = 'sprt_decision_96h' if 'sprt_decision_96h' in df.columns else 'sprt_decision_24h'
        has_labels = 'is_defect' in df.columns

        n_total = len(df)
        n_accept = (df[dec_col] == 'ACCEPT').sum()
        n_reject = (df[dec_col] == 'REJECT').sum()
        n_uncertain = (df[dec_col] == 'UNCERTAIN').sum()
        n_review = (df[dec_col] == 'REVIEW').sum()

        hours_saved = int(df['sprt_hours_saved'].sum())
        n_continued_to_96h = (df.get('sprt_decision_24h', pd.Series()) == 'UNCERTAIN').sum()

        report = {
            'total_parts': n_total,
            'accept_24h': int(n_accept),
            'reject_24h_or_96h': int(n_reject),
            'uncertain_continued_to_96h': int(n_continued_to_96h),
            'review_after_96h': int(n_review),
            'total_chamber_hours_saved': hours_saved,
        }

        if has_labels:
            y_true = df['is_defect'].values
            y_pred = df[dec_col].isin(['REJECT', 'REVIEW']).astype(int).values
            from sklearn.metrics import recall_score, precision_score, confusion_matrix
            report['recall'] = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
            report['precision'] = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            _, fp, fn, tp = cm.ravel()
            report['FN'] = int(fn)
            report['FP'] = int(fp)
            report['cost_50FN_1FP'] = int(50 * fn + fp)

            # Hours saved only for true defects
            true_reject = (df[dec_col].isin(['REJECT', 'REVIEW'])) & (df['is_defect'] == 1)
            report['true_defect_hours_saved'] = int(true_reject.sum() * 144)

        return report
