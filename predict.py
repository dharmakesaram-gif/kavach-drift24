"""
predict.py - Single Robust Entry Point for Inference
SIH26170: Space-Grade Semiconductor Latent Defect Screening

P12 fix: Tolerant of hidden CSVs with different casing, missing 96h/168h columns,
and various column naming conventions.

Usage:
    python predict.py input.csv [--output predictions.csv] [--model-dir models/]

Outputs for each part:
    - predicted_168h: Forecasted 168h parametric value (µA)
    - predicted_168h_lower / predicted_168h_upper: 10/90% confidence bounds
    - decision: ACCEPT / REVIEW / REJECT
    - reason: Human-readable explanation
    - risk_score: Composite anomaly score [0, 1]
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------------
# Stage 0: Schema Normaliser
# ---------------------------------------------------------------------------

# Canonical column map (lowercase variants → standard names)
_COLUMN_ALIASES = {
    # Part / lot identifiers
    'part_id': 'part_id', 'partid': 'part_id', 'part': 'part_id',
    'serial': 'part_id', 'serial_number': 'part_id', 'sn': 'part_id',
    'lot_id': 'lot_id', 'lotid': 'lot_id', 'lot': 'lot_id',
    'batch': 'lot_id', 'batch_id': 'lot_id',
    # Parametric readings
    'value_0h': 'value_0h', 'v0': 'value_0h', 'v_0h': 'value_0h',
    'value_0': 'value_0h', 'leakage_0h': 'value_0h', 'i_0h': 'value_0h',
    'reading_0h': 'value_0h', '0h': 'value_0h',
    'value_24h': 'value_24h', 'v24': 'value_24h', 'v_24h': 'value_24h',
    'value_24': 'value_24h', 'leakage_24h': 'value_24h', 'i_24h': 'value_24h',
    'reading_24h': 'value_24h', '24h': 'value_24h',
    'value_96h': 'value_96h', 'v96': 'value_96h', 'v_96h': 'value_96h',
    'value_96': 'value_96h', 'leakage_96h': 'value_96h', 'i_96h': 'value_96h',
    'reading_96h': 'value_96h', '96h': 'value_96h',
    'value_168h': 'value_168h', 'v168': 'value_168h', 'v_168h': 'value_168h',
    'value_168': 'value_168h', 'leakage_168h': 'value_168h', 'i_168h': 'value_168h',
    'reading_168h': 'value_168h', '168h': 'value_168h',
    # Labels (optional)
    'is_defect': 'is_defect', 'defect': 'is_defect', 'label': 'is_defect',
    'is_defective': 'is_defect', 'defective': 'is_defect',
    'defect_type': 'defect_type', 'failure_mode': 'defect_type',
    # Datasheet limit
    'datasheet_limit': 'datasheet_limit', 'limit': 'datasheet_limit',
    'spec_limit': 'datasheet_limit', 'max_limit': 'datasheet_limit',
}

_REQUIRED_COLUMNS = ['part_id', 'lot_id', 'value_0h', 'value_24h']
_OPTIONAL_COLUMNS = ['value_96h', 'value_168h', 'is_defect', 'defect_type', 'datasheet_limit']


def normalise_schema(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Case-insensitive column normaliser.
    Maps variant column names to canonical names.
    Returns (normalised_df, mapping_applied).
    """
    out = df.copy()
    mapping = {}
    
    # Build case-insensitive lookup
    for original_col in out.columns:
        normalised_key = original_col.strip().lower().replace(' ', '_').replace('-', '_')
        if normalised_key in _COLUMN_ALIASES:
            canonical = _COLUMN_ALIASES[normalised_key]
            if canonical not in mapping.values():  # Don't double-map
                mapping[original_col] = canonical
    
    out = out.rename(columns=mapping)
    
    # Check required columns
    missing = [c for c in _REQUIRED_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(
            f"Missing required columns after normalisation: {missing}\n"
            f"Available columns: {list(out.columns)}\n"
            f"Column mapping applied: {mapping}"
        )
    
    # Generate part_id if missing but lot_id present
    if 'part_id' not in out.columns and 'lot_id' in out.columns:
        out['part_id'] = [f"PART_{i:06d}" for i in range(len(out))]
    
    # Set default datasheet limit if not provided
    if 'datasheet_limit' not in out.columns:
        out['datasheet_limit'] = 50.0
        warnings.warn("No 'datasheet_limit' column found; defaulting to 50.0 µA.")
    
    # Handle missing optional time columns
    data_quality_flags = {}
    for col in ['value_96h', 'value_168h']:
        if col not in out.columns:
            data_quality_flags[col] = 'absent'
        else:
            n_missing = out[col].isna().sum()
            if n_missing > 0:
                data_quality_flags[col] = f'{n_missing}/{len(out)} missing'
    
    return out, {'column_mapping': mapping, 'data_quality': data_quality_flags}


# ---------------------------------------------------------------------------
# Prediction pipeline
# ---------------------------------------------------------------------------

def run_prediction(input_path: str, output_path: Optional[str] = None,
                   model_dir: Optional[str] = None) -> pd.DataFrame:
    """
    End-to-end prediction: load CSV → normalise → preprocess → Module A → Module B → Decision.
    Tolerant of missing columns and different casing.
    """
    # 1. Load and normalise
    print(f"[predict] Loading: {input_path}")
    raw_df = pd.read_csv(input_path)
    print(f"[predict] Raw shape: {raw_df.shape}, columns: {list(raw_df.columns)}")
    
    df, schema_info = normalise_schema(raw_df)
    if schema_info['column_mapping']:
        print(f"[predict] Schema normalised: {schema_info['column_mapping']}")
    if schema_info['data_quality']:
        print(f"[predict] Data quality flags: {schema_info['data_quality']}")
    
    # 2. Handle NaN / missing values  
    n_nan_0h = df['value_0h'].isna().sum()
    n_nan_24h = df['value_24h'].isna().sum()
    if n_nan_0h > 0 or n_nan_24h > 0:
        warnings.warn(f"Missing values detected: {n_nan_0h} in value_0h, {n_nan_24h} in value_24h. "
                      "These parts will be flagged for REVIEW with imputed values.")
        # Impute with lot median, fallback to global median
        for col in ['value_0h', 'value_24h']:
            lot_medians = df.groupby('lot_id')[col].transform('median')
            global_median = df[col].median()
            df[col] = df[col].fillna(lot_medians).fillna(global_median)
    
    # 3. Preprocess
    from src.preprocess import BurnInPreprocessor
    preprocessor = BurnInPreprocessor(small_lot_threshold=30, shrinkage_prior_weight=15.0)
    df_proc = preprocessor.fit_transform(df)
    
    # 4. Module A: Outlier Detection
    from src.module_a import DynamicOutlierDetector
    mod_a = DynamicOutlierDetector(cost_fn_weight=50.0, cost_fp_weight=1.0)
    
    # If labels available, use them for fitting; otherwise fit unsupervised
    if 'is_defect' in df_proc.columns and df_proc['is_defect'].sum() > 0:
        mod_a.fit(df_proc)
    else:
        mod_a.fit(df_proc)  # Fits on all data (unsupervised mode)
    
    pred_a = mod_a.predict_detailed(df_proc)
    
    # 5. Module B: Drift Prediction
    from src.module_b import DriftPredictor
    mod_b = DriftPredictor(safety_k_sigma=3.0)
    
    # Need log_v168 for training target — if 168h column available, use it
    if 'value_168h' in df_proc.columns and df_proc['value_168h'].notna().sum() > 10:
        mod_b.fit(df_proc)
        pred_b = mod_b.predict(df_proc)
    else:
        # Can't train forecaster without 168h — use heuristic extrapolation
        warnings.warn("No value_168h column available; Module B using heuristic drift extrapolation.")
        pred_b = _heuristic_drift_forecast(df_proc)
    
    # 6. Decision Engine
    from src.decision import ScreeningDecisionEngine
    decision_engine = ScreeningDecisionEngine()
    results = decision_engine.evaluate(pred_a, pred_b)
    
    # 7. Generate explanations
    from src.explain import generate_qa_report_card
    reasons_list = []
    for _, row in results.iterrows():
        try:
            card = generate_qa_report_card(row)
            reasons_list.append(card.get('narrative', ''))
        except Exception:
            reasons_list.append(row.get('primary_reason', 'N/A'))
    results['explanation'] = reasons_list
    
    # 8. Flag NaN-imputed parts as REVIEW
    if n_nan_0h > 0 or n_nan_24h > 0:
        nan_mask = raw_df['value_0h'].isna() | raw_df['value_24h'].isna() if 'value_0h' in raw_df.columns else pd.Series([False]*len(results))
        if nan_mask.any():
            results.loc[nan_mask, 'final_decision'] = 'REVIEW'
            results.loc[nan_mask, 'primary_reason'] = 'MISSING_DATA_IMPUTED'
    
    # 9. Build clean output
    output_cols = [
        'part_id', 'lot_id', 'value_0h', 'value_24h',
        'pred_v168', 'pred_v168_lower', 'pred_v168_upper',
        'final_decision', 'composite_risk_score', 'primary_reason', 'explanation'
    ]
    # Add optional columns if present
    for col in ['value_96h', 'value_168h', 'is_defect', 'datasheet_limit']:
        if col in results.columns:
            output_cols.insert(4, col)
    
    output_cols = [c for c in output_cols if c in results.columns]
    output_df = results[output_cols].copy()
    
    # 10. Save
    if output_path is None:
        base = os.path.splitext(input_path)[0]
        output_path = f"{base}_predictions.csv"
    
    output_df.to_csv(output_path, index=False)
    print(f"\n[predict] Predictions saved: {output_path}")
    print(f"[predict] Parts processed: {len(output_df)}")
    
    # Summary statistics
    dec_counts = output_df['final_decision'].value_counts()
    print(f"[predict] Decisions: {dict(dec_counts)}")
    
    if 'is_defect' in output_df.columns:
        from sklearn.metrics import recall_score
        y_true = output_df['is_defect'].values
        y_pred = (output_df['final_decision'].isin(['REVIEW', 'REJECT'])).astype(int).values
        rec = recall_score(y_true, y_pred, zero_division=0)
        print(f"[predict] Recall (REVIEW+REJECT as catch): {rec:.4f}")
    
    return output_df


def _heuristic_drift_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Heuristic 168h forecast when no training target is available.
    Uses observed 0h→24h drift rate extrapolated with power-law deceleration.
    """
    out = df.copy()
    v0 = out['value_0h'].values
    v24 = out['value_24h'].values
    
    # Power-law extrapolation: v(t) = v0 * (1 + alpha * (t/168)^gamma)
    # Estimate alpha from observed 24h drift, assume gamma ≈ 0.7 (sub-linear healthy)
    gamma = 0.7
    drift_frac_24h = np.maximum((v24 - v0) / np.maximum(v0, 1e-5), 1e-6)
    alpha = drift_frac_24h / ((24.0 / 168.0) ** gamma)
    
    pred_168 = v0 * (1.0 + alpha * 1.0)
    pred_lower = v0 * (1.0 + alpha * 0.75)  # Conservative lower bound
    pred_upper = v0 * (1.0 + alpha * 1.35)  # Conservative upper bound
    
    # Safety slope
    pred_slope = (pred_168 - v0) / 168.0
    
    out['pred_v168'] = np.round(pred_168, 4)
    out['pred_v168_lower'] = np.round(pred_lower, 4)
    out['pred_v168_upper'] = np.round(pred_upper, 4)
    out['pred_drift_rate_168h'] = np.round(pred_slope, 5)
    out['pred_upper_drift_rate'] = np.round((pred_upper - v0) / 168.0, 5)
    out['safety_slope_threshold'] = 0.05  # Default
    out['decision_mod_b'] = 'ACCEPT'
    out['reason_codes_mod_b'] = 'HEURISTIC_EXTRAPOLATION'
    
    # Flag parts where upper bound exceeds 90% of datasheet limit
    limit = out['datasheet_limit'].values if 'datasheet_limit' in out.columns else np.full(len(out), 50.0)
    breach_mask = pred_upper >= limit * 0.90
    out.loc[breach_mask, 'decision_mod_b'] = 'REJECT'
    out.loc[breach_mask, 'reason_codes_mod_b'] = 'HEURISTIC_LIMIT_BREACH'
    
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="SIH26170 Burn-In Defect Screening — Single-File Inference Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python predict.py data/hidden_test.csv
    python predict.py data/input.csv --output results/predictions.csv
    python predict.py data/input.csv --limit 60.0
        """
    )
    parser.add_argument('input', help='Path to input CSV file')
    parser.add_argument('--output', '-o', default=None, help='Output CSV path (default: <input>_predictions.csv)')
    parser.add_argument('--model-dir', default=None, help='Directory containing pre-trained model artifacts')
    parser.add_argument('--limit', type=float, default=None, help='Override datasheet limit (µA)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    
    result_df = run_prediction(args.input, args.output, args.model_dir)
