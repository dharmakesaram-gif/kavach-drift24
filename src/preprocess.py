"""
preprocess.py - Robust Preprocessing & Dynamic Part Average Testing (DPAT)
SIH26170: Space-Grade Semiconductor Latent Defect Screening

Features:
- Natural log transformations for positive parametric measurements
- Robust Per-Lot Normalization using Median and MAD (AEC-Q001 Standard)
- Empirical Bayes shrinkage for small lots (< 30 parts) to prevent overfitting
- Early drift velocity and acceleration features (0h -> 24h)
- Lot-relative drift deviation metrics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any


class BurnInPreprocessor:
    """
    Stateful preprocessor for semiconductor burn-in time-series data.
    Fits robust lot statistics on historical/training data with shrinkage fallback,
    and transforms new production lots.
    """
    
    def __init__(self, small_lot_threshold: int = 30, shrinkage_prior_weight: float = 15.0):
        self.small_lot_threshold = small_lot_threshold
        self.shrinkage_prior_weight = shrinkage_prior_weight
        self.fitted = False
        
        # Pooled global baseline statistics
        self.global_stats: Dict[str, Dict[str, float]] = {}
        # Per-lot statistics
        self.lot_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        
    @staticmethod
    def calc_mad(series: pd.Series) -> float:
        """Computes Median Absolute Deviation (MAD), avoiding zero division with floor."""
        med = np.median(series)
        mad = np.median(np.abs(series - med))
        # 1.4826 scales MAD to be an unbiased estimator of standard deviation for normal dist
        scale = 1.4826 * mad
        return max(float(scale), 1e-4)

    def fit(self, df: pd.DataFrame) -> 'BurnInPreprocessor':
        """
        Computes pooled global and per-lot baseline parameters for 0h and 24h measurements and drift.
        Applies Empirical Bayes shrinkage only for lots smaller than small_lot_threshold.
        """
        temp_df = df.copy()
        temp_df['log_v0'] = np.log(np.maximum(temp_df['value_0h'], 1e-5))
        temp_df['log_v24'] = np.log(np.maximum(temp_df['value_24h'], 1e-5))
        temp_df['drift_24h'] = temp_df['log_v24'] - temp_df['log_v0']
        temp_df['rel_drift_24h'] = (temp_df['value_24h'] - temp_df['value_0h']) / np.maximum(temp_df['value_0h'], 1e-5)
        
        cols_to_stat = ['value_0h', 'value_24h', 'log_v0', 'log_v24', 'drift_24h', 'rel_drift_24h']
        
        # Calculate global pooled priors
        self.global_stats = {}
        for col in cols_to_stat:
            series = temp_df[col].dropna()
            med = float(np.median(series))
            mad_scale = self.calc_mad(series)
            self.global_stats[col] = {'median': med, 'mad_scale': mad_scale}
            
        # Calculate per-lot statistics — apply shrinkage ONLY for small lots
        self.lot_stats = {}
        grouped = temp_df.groupby('lot_id')
        for lot_id, lot_group in grouped:
            n_parts = len(lot_group)
            self.lot_stats[lot_id] = {'count': n_parts}
            
            # FIX: Only shrink small lots toward global prior
            if n_parts < self.small_lot_threshold:
                lamb = n_parts / (n_parts + self.shrinkage_prior_weight)
            else:
                lamb = 1.0  # Large lots use their own stats without shrinkage
            
            for col in cols_to_stat:
                l_series = lot_group[col].dropna()
                if len(l_series) > 0:
                    l_med = float(np.median(l_series))
                    l_mad = self.calc_mad(l_series)
                    
                    # Shrink toward global pooled distribution only for small lots
                    g_med = self.global_stats[col]['median']
                    g_mad = self.global_stats[col]['mad_scale']
                    
                    shrunk_med = lamb * l_med + (1.0 - lamb) * g_med
                    shrunk_mad = lamb * l_mad + (1.0 - lamb) * g_mad
                    
                    self.lot_stats[lot_id][col] = {
                        'median': shrunk_med,
                        'mad_scale': max(shrunk_mad, 1e-4)
                    }
                else:
                    self.lot_stats[lot_id][col] = self.global_stats[col]
                    
        self.fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw burn-in parametric readings into lot-normalized, drift, and z-score features.
        Uses vectorised groupby/map instead of iterrows for performance.
        """
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted on training data before calling transform!")
            
        out = df.copy()
        
        # 1. Primary Log Transforms
        out['log_v0'] = np.log(np.maximum(out['value_0h'], 1e-5))
        out['log_v24'] = np.log(np.maximum(out['value_24h'], 1e-5))
        
        if 'value_96h' in out.columns:
            out['log_v96'] = np.log(np.maximum(out['value_96h'], 1e-5))
        if 'value_168h' in out.columns:
            out['log_v168'] = np.log(np.maximum(out['value_168h'], 1e-5))
            
        # 2. Physics-Informed Drift Features
        out['delta_24h'] = out['log_v24'] - out['log_v0']
        out['slope_24h'] = (out['value_24h'] - out['value_0h']) / 24.0  # µA/hour
        out['rel_drift_24h'] = (out['value_24h'] - out['value_0h']) / np.maximum(out['value_0h'], 1e-5)
        out['ratio_24h'] = out['value_24h'] / np.maximum(out['value_0h'], 1e-5)
        
        # 3. Dynamic Part Average Testing (DPAT) per lot — VECTORISED
        # Compute batch stats for unseen lots using groupby instead of row-loop
        df_lots = out['lot_id'].unique()
        unseen_lots = [l for l in df_lots if l not in self.lot_stats]
        
        # Build batch stats for unseen lots from current data
        batch_lot_stats = {}
        if unseen_lots:
            unseen_mask = out['lot_id'].isin(unseen_lots)
            unseen_df = out[unseen_mask]
            for l_id, grp in unseen_df.groupby('lot_id'):
                n = len(grp)
                if n < self.small_lot_threshold:
                    lamb = n / (n + self.shrinkage_prior_weight)
                else:
                    lamb = 1.0
                batch_lot_stats[l_id] = {}
                for col, src_col in [('value_0h', 'value_0h'), ('value_24h', 'value_24h'), ('drift_24h', 'delta_24h')]:
                    s = grp[src_col].dropna()
                    if len(s) > 1:
                        m = float(np.median(s))
                        md = self.calc_mad(s)
                        gm = self.global_stats[col]['median']
                        gmd = self.global_stats[col]['mad_scale']
                        batch_lot_stats[l_id][col] = {
                            'median': lamb * m + (1 - lamb) * gm,
                            'mad_scale': max(lamb * md + (1 - lamb) * gmd, 1e-4)
                        }
                    else:
                        batch_lot_stats[l_id][col] = self.global_stats[col]

        # Vectorised stat lookup using map
        def _get_stat(lot_id, col, stat_key):
            src = self.lot_stats.get(lot_id, batch_lot_stats.get(lot_id, {}))
            entry = src.get(col, self.global_stats.get(col, {'median': 0.0, 'mad_scale': 1e-4}))
            return entry[stat_key]

        out['lot_median_v0'] = out['lot_id'].map(lambda l: _get_stat(l, 'value_0h', 'median'))
        out['lot_mad_v0'] = out['lot_id'].map(lambda l: _get_stat(l, 'value_0h', 'mad_scale'))
        out['lot_median_v24'] = out['lot_id'].map(lambda l: _get_stat(l, 'value_24h', 'median'))
        out['lot_mad_v24'] = out['lot_id'].map(lambda l: _get_stat(l, 'value_24h', 'mad_scale'))
        out['lot_median_drift_24h'] = out['lot_id'].map(lambda l: _get_stat(l, 'drift_24h', 'median'))
        out['lot_mad_drift_24h'] = out['lot_id'].map(lambda l: _get_stat(l, 'drift_24h', 'mad_scale'))
        
        # 4. Robust DPAT Z-scores (AEC-Q001 Standard): (Value - Median) / MAD_Scale
        out['z_pat_0h'] = (out['value_0h'] - out['lot_median_v0']) / out['lot_mad_v0']
        out['z_pat_24h'] = (out['value_24h'] - out['lot_median_v24']) / out['lot_mad_v24']
        out['z_drift_24h'] = (out['delta_24h'] - out['lot_median_drift_24h']) / out['lot_mad_drift_24h']
        
        # 5. Relative lot deviation ratios
        out['ratio_to_lot_v0'] = out['value_0h'] / np.maximum(out['lot_median_v0'], 1e-4)
        out['ratio_to_lot_24h'] = out['value_24h'] / np.maximum(out['lot_median_v24'], 1e-4)
        out['excess_slope_24h'] = out['delta_24h'] - out['lot_median_drift_24h']
        
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fits statistics and transforms the dataframe in one call."""
        return self.fit(df).transform(df)


def get_feature_columns() -> List[str]:
    """Returns list of normalized feature names for Module A and Module B."""
    return [
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
