"""
generate.py - Synthetic Burn-In Parametric Data Generator
SIH26170: Space-Grade Semiconductor Latent Defect Screening

Simulates multi-lot burn-in stress screening (125°C ESS) at intervals: 0h, 24h, 96h, and 168h.
Generates healthy parts with lot-to-lot baseline variations and 4 distinct latent defect mechanisms.
"""

import os
import argparse
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any


def generate_burnin_dataset(
    n_lots: int = 40,
    parts_per_lot_range: Tuple[int, int] = (150, 400),
    defect_rate: float = 0.015,
    datasheet_limit: float = 50.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Generates realistic semiconductor parametric burn-in time-series data.
    
    Parameters:
        n_lots: Number of semiconductor fabrication/assembly lots (e.g. 30-50).
        parts_per_lot_range: (min, max) parts per lot.
        defect_rate: Proportion of parts with latent anomalies (0.5% - 2.0%).
        datasheet_limit: Static pass/fail threshold in µA (e.g. 50.0 µA).
        random_state: Seed for reproducibility.
        
    Returns:
        pd.DataFrame containing:
          - part_id, lot_id
          - value_0h, value_24h, value_96h, value_168h (leakage current in µA)
          - is_defect (0=healthy, 1=defect)
          - defect_type ('healthy', 'high_offset', 'steep_linear', 'accelerating', 'subtle_in_spec')
          - datasheet_limit
    """
    rng = np.random.default_rng(random_state)
    
    rows = []
    global_part_counter = 1000
    
    # Defect type probabilities (conditional on being defective)
    defect_types = ['high_offset', 'steep_linear', 'accelerating', 'subtle_in_spec']
    defect_weights = [0.25, 0.25, 0.25, 0.25]
    
    for lot_idx in range(1, n_lots + 1):
        lot_id = f"LOT_{lot_idx:03d}"
        n_parts = rng.integers(parts_per_lot_range[0], parts_per_lot_range[1] + 1)
        
        # Each lot has intrinsic wafer fab variations:
        # Lot median leakage typically 6 to 18 µA (log-normal distribution)
        lot_median_base = float(rng.lognormal(mean=np.log(10.5), sigma=0.22))
        lot_sigma_rel = float(rng.uniform(0.08, 0.14))  # Relative standard deviation within lot
        
        for _ in range(n_parts):
            global_part_counter += 1
            part_id = f"SN_{global_part_counter:06d}"
            
            is_defective = rng.random() < defect_rate
            
            if not is_defective:
                defect_type = 'healthy'
                # Log-normal distribution within the lot
                v0 = float(rng.lognormal(mean=np.log(lot_median_base), sigma=lot_sigma_rel))
                # Ensure healthy part is strictly within normal lot band (< 2.5 sigma and well below limit)
                v0 = min(v0, datasheet_limit * 0.45)
                
                # Healthy drift: subtle physical aging at 125°C (Arrhenius/power-law saturation)
                # alpha: total drift fraction over 168h, typically 5% to 22%
                alpha = float(rng.uniform(0.05, 0.22))
                gamma = float(rng.uniform(0.65, 0.85))  # Sublinear saturation
                
                drift_24 = alpha * ((24.0 / 168.0) ** gamma)
                drift_96 = alpha * ((96.0 / 168.0) ** gamma)
                drift_168 = alpha * 1.0
                
                # Add small high-precision measurement noise (sigma ~ 0.5% - 1%)
                noise_24 = float(rng.normal(0, 0.008 * v0))
                noise_96 = float(rng.normal(0, 0.008 * v0))
                noise_168 = float(rng.normal(0, 0.008 * v0))
                
                v24 = v0 * (1.0 + drift_24) + noise_24
                v96 = v0 * (1.0 + drift_96) + noise_96
                v168 = v0 * (1.0 + drift_168) + noise_168
                
            else:
                # Part has a latent defect
                defect_type = rng.choice(defect_types, p=defect_weights)
                
                if defect_type == 'high_offset':
                    # Type 1: High initial offset (latent oxide pinhole / ESD damage)
                    # Starts elevated (3.5 to 6.5 sigma above lot median), but still passes static limit!
                    low_val = min(lot_median_base * 2.5, datasheet_limit * 0.70)
                    high_val = max(low_val + 1.0, min(datasheet_limit * 0.92, lot_median_base * 4.2))
                    v0 = float(rng.uniform(low_val, high_val))
                    alpha = float(rng.uniform(0.12, 0.35))
                    v24 = v0 * (1.0 + alpha * (24.0 / 168.0)) + float(rng.normal(0, 0.01 * v0))
                    v96 = v0 * (1.0 + alpha * (96.0 / 168.0)) + float(rng.normal(0, 0.01 * v0))
                    v168 = v0 * (1.0 + alpha) + float(rng.normal(0, 0.01 * v0))
                    
                elif defect_type == 'steep_linear':
                    # Type 2: Steep linear drift (progressive electromigration / mobile ion migration)
                    # Starts in the normal lot band at 0h, but drifts 5x-15x faster than healthy parts
                    v0 = float(rng.lognormal(mean=np.log(lot_median_base), sigma=lot_sigma_rel))
                    slope_factor = float(rng.uniform(1.2, 3.2))  # 120% to 320% total drift over 168h
                    v24 = v0 * (1.0 + slope_factor * (24.0 / 168.0)) + float(rng.normal(0, 0.01 * v0))
                    v96 = v0 * (1.0 + slope_factor * (96.0 / 168.0)) + float(rng.normal(0, 0.015 * v0))
                    v168 = v0 * (1.0 + slope_factor) + float(rng.normal(0, 0.02 * v0))
                    
                elif defect_type == 'accelerating':
                    # Type 3: Accelerating drift (Time-Dependent Dielectric Breakdown - TDDB)
                    # Normal at 0h, subtle at 24h (looks almost normal!), then accelerates exponentially
                    v0 = float(rng.lognormal(mean=np.log(lot_median_base), sigma=lot_sigma_rel))
                    # Early subtle rise at 24h
                    early_boost = float(rng.uniform(0.15, 0.35))
                    v24 = v0 * (1.0 + early_boost * (24.0 / 168.0) + float(rng.normal(0, 0.01 * v0)))
                    # Superlinear exponent
                    p_exp = float(rng.uniform(2.1, 2.8))
                    v96 = v0 * (1.0 + 2.5 * ((96.0 / 168.0) ** p_exp)) + float(rng.normal(0, 0.02 * v0))
                    v168 = v0 * (1.0 + 2.5 * (1.0 ** p_exp)) + float(rng.normal(0, 0.03 * v0))
                    
                elif defect_type == 'subtle_in_spec':
                    # Type 4: Subtle in-spec defect (The classic SIH prompt example:
                    # Lot mean = 10 µA, part is at 45 µA, datasheet limit = 50 µA)
                    # Static screening marks PASS, but it is 4.5x lot mean!
                    target_ratio = float(rng.uniform(3.2, 4.4))
                    v0 = min(lot_median_base * target_ratio, datasheet_limit * 0.88)
                    v24 = v0 * (1.0 + float(rng.uniform(0.10, 0.25)) * (24.0 / 168.0))
                    v96 = v0 * (1.0 + float(rng.uniform(0.20, 0.40)) * (96.0 / 168.0))
                    v168 = v0 * (1.0 + float(rng.uniform(0.35, 0.65)))
            
            # Ensure physical lower bound (leakage > 0)
            v0 = max(0.5, round(v0, 4))
            v24 = max(0.5, round(v24, 4))
            v96 = max(0.5, round(v96, 4))
            v168 = max(0.5, round(v168, 4))
            
            # Static screening decision (for baseline comparison)
            static_pass = (v0 <= datasheet_limit) and (v24 <= datasheet_limit) and \
                          (v96 <= datasheet_limit) and (v168 <= datasheet_limit)
            
            rows.append({
                'part_id': part_id,
                'lot_id': lot_id,
                'value_0h': v0,
                'value_24h': v24,
                'value_96h': v96,
                'value_168h': v168,
                'is_defect': 1 if is_defective else 0,
                'defect_type': defect_type,
                'datasheet_limit': datasheet_limit,
                'static_pass': 1 if static_pass else 0
            })
            
    df = pd.DataFrame(rows)
    return df


def split_lots(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Performs Lot-Grouped Splitting to ensure zero data leakage between production lots.
    """
    rng = np.random.default_rng(random_state)
    lots = np.sort(df['lot_id'].unique())
    rng.shuffle(lots)
    
    n_lots = len(lots)
    n_train = int(n_lots * train_ratio)
    n_val = int(n_lots * val_ratio)
    
    train_lots = lots[:n_train]
    val_lots = lots[n_train:n_train + n_val]
    test_lots = lots[n_train + n_val:]
    
    df_train = df[df['lot_id'].isin(train_lots)].copy().reset_index(drop=True)
    df_val = df[df['lot_id'].isin(val_lots)].copy().reset_index(drop=True)
    df_test = df[df['lot_id'].isin(test_lots)].copy().reset_index(drop=True)
    
    return df_train, df_val, df_test


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate synthetic burn-in test dataset")
    parser.add_argument("--n_lots", type=int, default=40, help="Number of wafer lots")
    parser.add_argument("--output_dir", type=str, default="data/synthetic", help="Output directory")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("data/raw", exist_ok=True)
    
    print(f"Generating burn-in dataset with {args.n_lots} lots...")
    df = generate_burnin_dataset(n_lots=args.n_lots, random_state=42)
    
    full_path = os.path.join(args.output_dir, "burnin_full_dataset.csv")
    df.to_csv(full_path, index=False)
    print(f"Saved full dataset: {full_path} ({len(df)} parts across {df['lot_id'].nunique()} lots)")
    
    train_df, val_df, test_df = split_lots(df, train_ratio=0.70, val_ratio=0.15, random_state=42)
    
    train_path = os.path.join(args.output_dir, "burnin_train.csv")
    val_path = os.path.join(args.output_dir, "burnin_val.csv")
    test_path = os.path.join(args.output_dir, "burnin_test.csv")
    test_blind_path = os.path.join(args.output_dir, "burnin_test_blind.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Blind test set: hide 168h value for blind evaluation of Module B
    test_blind_df = test_df.drop(columns=['value_168h']).copy()
    test_blind_df.to_csv(test_blind_path, index=False)
    
    print(f"  Train set: {len(train_df)} parts ({train_df['is_defect'].sum()} defects, {train_df['lot_id'].nunique()} lots)")
    print(f"  Val set:   {len(val_df)} parts ({val_df['is_defect'].sum()} defects, {val_df['lot_id'].nunique()} lots)")
    print(f"  Test set:  {len(test_df)} parts ({test_df['is_defect'].sum()} defects, {test_df['lot_id'].nunique()} lots)")
    print(f"  Blind set: {len(test_blind_df)} parts saved to {test_blind_path}")
