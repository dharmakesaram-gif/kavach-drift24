"""
generate_physics.py - Physics-Based Synthetic Data Generator
SIH26170: Space-Grade Semiconductor Latent Defect Screening

Phase 1 upgrade: Replaces convenient distributions with published failure-mechanism
models (JEDEC JEP122, JESD22-A108). Features:
- 6 physics-based failure mechanisms with literature-grounded parameters
- Continuous defect severity (some defects nearly healthy)
- Tester repeatability noise, temperature offset, missing readings, heavy-tail lots
- Held-out generator configuration for domain-shift testing
- Stress-test suite scenarios (maverick lots, tiny lots, noise amplification, etc.)
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field


@dataclass
class GeneratorConfig:
    """Configuration for the physics-based generator. Vary for held-out testing."""
    n_lots: int = 45
    parts_per_lot_range: Tuple[int, int] = (150, 400)
    defect_rate: float = 0.015
    datasheet_limit: float = 50.0
    random_state: int = 42
    
    # Lot-to-lot variation
    lot_median_log_mean: float = np.log(10.5)
    lot_median_log_sigma: float = 0.22
    lot_sigma_rel_range: Tuple[float, float] = (0.08, 0.14)
    
    # Tester noise
    tester_noise_sigma: float = 0.008  # Relative to value
    temperature_offset_sigma: float = 0.003  # Systematic per-readout offset
    missing_rate: float = 0.0  # Fraction of readings set to NaN
    
    # Heavy-tail lot probability
    heavy_tail_lot_fraction: float = 0.0  # Fraction of lots with heavy-tail distribution
    
    # Defect mechanism weights
    defect_weights: Dict[str, float] = field(default_factory=lambda: {
        'nbti_ageing': 0.20,
        'tddb_dielectric': 0.20,
        'electromigration': 0.15,
        'mobile_ion': 0.15,
        'esd_oxide': 0.15,
        'subtle_in_spec': 0.15,
    })
    
    # Defect severity: Beta distribution parameters (lower alpha → more near-healthy)
    severity_alpha: float = 1.5
    severity_beta: float = 3.0


# ---------------------------------------------------------------------------
# Default and held-out configurations
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = GeneratorConfig()

HELD_OUT_CONFIG = GeneratorConfig(
    random_state=9999,
    lot_median_log_sigma=0.30,          # Wider lot spread
    lot_sigma_rel_range=(0.10, 0.18),   # More within-lot variation
    defect_rate=0.025,                  # Higher defect rate
    tester_noise_sigma=0.012,           # Noisier tester
    temperature_offset_sigma=0.005,
    missing_rate=0.01,
    heavy_tail_lot_fraction=0.15,
    severity_alpha=1.2,                 # More near-healthy defects
    severity_beta=2.5,
    defect_weights={
        'nbti_ageing': 0.25,
        'tddb_dielectric': 0.25,
        'electromigration': 0.15,
        'mobile_ion': 0.10,
        'esd_oxide': 0.10,
        'subtle_in_spec': 0.15,
    }
)


# ---------------------------------------------------------------------------
# Stress-test scenario overrides
# ---------------------------------------------------------------------------

def maverick_lot_config() -> GeneratorConfig:
    """35% contaminated lot scenario."""
    cfg = GeneratorConfig(n_lots=10, random_state=7777)
    cfg.defect_rate = 0.35
    return cfg

def tiny_lot_config() -> GeneratorConfig:
    """15-20 parts per lot."""
    cfg = GeneratorConfig(n_lots=20, random_state=8888)
    cfg.parts_per_lot_range = (15, 20)
    return cfg

def noisy_tester_config() -> GeneratorConfig:
    """Tester repeatability 3× worse."""
    cfg = GeneratorConfig(random_state=6666)
    cfg.tester_noise_sigma = 0.024
    cfg.temperature_offset_sigma = 0.009
    return cfg

def missing_data_config() -> GeneratorConfig:
    """5% missing readings in 0h and 24h."""
    cfg = GeneratorConfig(random_state=5555)
    cfg.missing_rate = 0.05
    return cfg

def heavy_tail_config() -> GeneratorConfig:
    """30% of lots have heavy-tail distributions."""
    cfg = GeneratorConfig(random_state=4444)
    cfg.heavy_tail_lot_fraction = 0.30
    return cfg


STRESS_CONFIGS = {
    'maverick_lot': maverick_lot_config(),
    'tiny_lot': tiny_lot_config(),
    'noisy_tester': noisy_tester_config(),
    'missing_data': missing_data_config(),
    'heavy_tail': heavy_tail_config(),
}


# ---------------------------------------------------------------------------
# Physics-based mechanism models
# ---------------------------------------------------------------------------

def _healthy_ageing(rng: np.random.Generator, v0: float, lot_sigma: float,
                    noise_sigma: float, temp_offset_sigma: float) -> Tuple[float, float, float]:
    """
    NBTI-like healthy ageing: power-law drift ΔV ∝ t^n, n ≈ 0.15-0.3.
    Leakage drift saturates sub-linearly.
    """
    n_exp = rng.uniform(0.15, 0.30)  # Power-law exponent (JEDEC typical)
    alpha = rng.uniform(0.05, 0.20)  # Total fractional drift at 168h
    
    drift_24 = alpha * (24.0 / 168.0) ** n_exp
    drift_96 = alpha * (96.0 / 168.0) ** n_exp
    drift_168 = alpha  # normalised to 1.0 at 168h
    
    # Tester noise + temperature offset at each readout
    temp_offset = rng.normal(0, temp_offset_sigma * v0)
    v24 = v0 * (1.0 + drift_24) + rng.normal(0, noise_sigma * v0) + temp_offset
    v96 = v0 * (1.0 + drift_96) + rng.normal(0, noise_sigma * v0) + temp_offset * 0.5
    v168 = v0 * (1.0 + drift_168) + rng.normal(0, noise_sigma * v0)
    
    return v24, v96, v168


def _nbti_defect(rng: np.random.Generator, v0: float, severity: float,
                 noise_sigma: float) -> Tuple[float, float, float]:
    """
    NBTI-like defect: Accelerated power-law drift with higher exponent.
    Severity controls how much faster than healthy.
    """
    n_exp = 0.25 + severity * 0.25  # Higher exponent for defects
    alpha = 0.20 + severity * 1.5   # Much larger fractional drift
    
    drift_24 = alpha * (24.0 / 168.0) ** n_exp
    drift_96 = alpha * (96.0 / 168.0) ** n_exp
    drift_168 = alpha
    
    v24 = v0 * (1.0 + drift_24) + rng.normal(0, noise_sigma * v0)
    v96 = v0 * (1.0 + drift_96) + rng.normal(0, noise_sigma * v0)
    v168 = v0 * (1.0 + drift_168) + rng.normal(0, noise_sigma * v0)
    
    return v24, v96, v168


def _tddb_defect(rng: np.random.Generator, v0: float, severity: float,
                 noise_sigma: float) -> Tuple[float, float, float]:
    """
    TDDB (Time-Dependent Dielectric Breakdown): Looks almost normal at 24h,
    then accelerates (E-model / power-law time dependence).
    """
    # Early subtle rise at 24h
    early_factor = 0.05 + severity * 0.15
    # Super-linear acceleration exponent
    p_exp = 1.8 + severity * 1.2  # 1.8 to 3.0
    total_drift = 0.5 + severity * 3.0
    
    drift_24 = early_factor * (24.0 / 168.0)
    drift_96 = total_drift * (96.0 / 168.0) ** p_exp
    drift_168 = total_drift
    
    v24 = v0 * (1.0 + drift_24) + rng.normal(0, noise_sigma * v0)
    v96 = v0 * (1.0 + drift_96) + rng.normal(0, noise_sigma * v0 * 1.5)
    v168 = v0 * (1.0 + drift_168) + rng.normal(0, noise_sigma * v0 * 2.0)
    
    return v24, v96, v168


def _electromigration_defect(rng: np.random.Generator, v0: float, severity: float,
                              noise_sigma: float) -> Tuple[float, float, float]:
    """
    Electromigration (Black's equation): Steady super-healthy drift rate.
    MTTF = A · J^(-n) · exp(Ea/kT), manifests as constant-rate leakage rise.
    """
    slope_factor = 0.8 + severity * 2.5  # Linear slope multiplier
    
    v24 = v0 * (1.0 + slope_factor * 24.0 / 168.0) + rng.normal(0, noise_sigma * v0)
    v96 = v0 * (1.0 + slope_factor * 96.0 / 168.0) + rng.normal(0, noise_sigma * v0 * 1.2)
    v168 = v0 * (1.0 + slope_factor) + rng.normal(0, noise_sigma * v0 * 1.5)
    
    return v24, v96, v168


def _mobile_ion_defect(rng: np.random.Generator, v0: float, severity: float,
                       noise_sigma: float) -> Tuple[float, float, float]:
    """
    Mobile-ion contamination: Bias-temperature dependent threshold shift.
    Early step then saturation.
    """
    step_time = rng.uniform(10, 50)  # Hours when step occurs
    step_magnitude = (0.3 + severity * 2.0) * v0  # Step size in µA
    saturation_rate = rng.uniform(0.01, 0.05)  # Post-step slow saturation
    
    def _value_at(t):
        if t < step_time:
            return v0 + step_magnitude * (t / step_time) ** 0.5
        else:
            post_step = v0 + step_magnitude
            return post_step * (1.0 + saturation_rate * (1 - np.exp(-(t - step_time) / 100.0)))
    
    v24 = _value_at(24.0) + rng.normal(0, noise_sigma * v0)
    v96 = _value_at(96.0) + rng.normal(0, noise_sigma * v0)
    v168 = _value_at(168.0) + rng.normal(0, noise_sigma * v0)
    
    return v24, v96, v168


def _esd_oxide_defect(rng: np.random.Generator, v0_lot_median: float, severity: float,
                      noise_sigma: float, limit: float) -> Tuple[float, float, float, float]:
    """
    ESD / oxide damage: High initial offset (starts elevated), moderate drift.
    Returns (v0, v24, v96, v168) — v0 is elevated.
    """
    # Elevated initial value: 2.5× to 4.5× lot median, still below limit
    offset_factor = 2.5 + severity * 2.0
    v0 = min(v0_lot_median * offset_factor, limit * 0.92)
    
    alpha = rng.uniform(0.10, 0.25 + severity * 0.15)
    v24 = v0 * (1.0 + alpha * 24.0 / 168.0) + rng.normal(0, noise_sigma * v0)
    v96 = v0 * (1.0 + alpha * 96.0 / 168.0) + rng.normal(0, noise_sigma * v0)
    v168 = v0 * (1.0 + alpha) + rng.normal(0, noise_sigma * v0)
    
    return v0, v24, v96, v168


def _subtle_in_spec_defect(rng: np.random.Generator, v0_lot_median: float, severity: float,
                            noise_sigma: float, limit: float) -> Tuple[float, float, float, float]:
    """
    Subtle in-spec defect: The classic SIH26170 problem statement example.
    Part at 45 µA with lot mean 10 µA, datasheet limit 50 µA → static PASS but defective.
    """
    target_ratio = 2.5 + severity * 2.5  # 2.5× to 5× lot median
    v0 = min(v0_lot_median * target_ratio, limit * 0.88)
    
    drift = 0.10 + severity * 0.40
    v24 = v0 * (1.0 + drift * 24.0 / 168.0) + rng.normal(0, noise_sigma * v0)
    v96 = v0 * (1.0 + drift * 96.0 / 168.0) + rng.normal(0, noise_sigma * v0)
    v168 = v0 * (1.0 + drift) + rng.normal(0, noise_sigma * v0)
    
    return v0, v24, v96, v168


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_physics_dataset(cfg: Optional[GeneratorConfig] = None) -> pd.DataFrame:
    """
    Generates semiconductor burn-in data using physics-based failure mechanisms.
    """
    if cfg is None:
        cfg = DEFAULT_CONFIG
    
    rng = np.random.default_rng(cfg.random_state)
    rows = []
    part_counter = 1000
    
    mechanisms = list(cfg.defect_weights.keys())
    mech_weights = [cfg.defect_weights[m] for m in mechanisms]
    
    for lot_idx in range(1, cfg.n_lots + 1):
        lot_id = f"LOT_{lot_idx:03d}"
        n_parts = rng.integers(cfg.parts_per_lot_range[0], cfg.parts_per_lot_range[1] + 1)
        
        # Lot-level parameters
        lot_median_base = float(rng.lognormal(mean=cfg.lot_median_log_mean, sigma=cfg.lot_median_log_sigma))
        
        # Heavy-tail lot variant
        is_heavy_tail = rng.random() < cfg.heavy_tail_lot_fraction
        if is_heavy_tail:
            lot_sigma_rel = float(rng.uniform(cfg.lot_sigma_rel_range[1], cfg.lot_sigma_rel_range[1] * 2.0))
        else:
            lot_sigma_rel = float(rng.uniform(*cfg.lot_sigma_rel_range))
        
        for _ in range(n_parts):
            part_counter += 1
            part_id = f"SN_{part_counter:06d}"
            is_defective = rng.random() < cfg.defect_rate
            
            if not is_defective:
                # Healthy part: log-normal within lot
                v0 = float(rng.lognormal(mean=np.log(lot_median_base), sigma=lot_sigma_rel))
                v0 = min(v0, cfg.datasheet_limit * 0.45)
                v24, v96, v168 = _healthy_ageing(rng, v0, lot_sigma_rel,
                                                  cfg.tester_noise_sigma, cfg.temperature_offset_sigma)
                defect_type = 'healthy'
                severity = 0.0
            else:
                # Draw continuous severity from Beta distribution
                severity = float(rng.beta(cfg.severity_alpha, cfg.severity_beta))
                severity = max(0.01, min(severity, 0.99))
                
                # Select failure mechanism
                defect_type = rng.choice(mechanisms, p=mech_weights)
                
                # Generate base v0 for in-lot-start mechanisms
                v0 = float(rng.lognormal(mean=np.log(lot_median_base), sigma=lot_sigma_rel))
                
                if defect_type == 'nbti_ageing':
                    v24, v96, v168 = _nbti_defect(rng, v0, severity, cfg.tester_noise_sigma)
                    
                elif defect_type == 'tddb_dielectric':
                    v24, v96, v168 = _tddb_defect(rng, v0, severity, cfg.tester_noise_sigma)
                    
                elif defect_type == 'electromigration':
                    v24, v96, v168 = _electromigration_defect(rng, v0, severity, cfg.tester_noise_sigma)
                    
                elif defect_type == 'mobile_ion':
                    v24, v96, v168 = _mobile_ion_defect(rng, v0, severity, cfg.tester_noise_sigma)
                    
                elif defect_type == 'esd_oxide':
                    v0, v24, v96, v168 = _esd_oxide_defect(rng, lot_median_base, severity,
                                                            cfg.tester_noise_sigma, cfg.datasheet_limit)
                    
                elif defect_type == 'subtle_in_spec':
                    v0, v24, v96, v168 = _subtle_in_spec_defect(rng, lot_median_base, severity,
                                                                 cfg.tester_noise_sigma, cfg.datasheet_limit)
                else:
                    v24, v96, v168 = _healthy_ageing(rng, v0, lot_sigma_rel,
                                                      cfg.tester_noise_sigma, cfg.temperature_offset_sigma)
            
            # Physical lower bound (leakage > 0)
            v0 = max(0.5, round(v0, 4))
            v24 = max(0.5, round(v24, 4))
            v96 = max(0.5, round(v96, 4))
            v168 = max(0.5, round(v168, 4))
            
            # Inject missing readings
            if cfg.missing_rate > 0:
                if rng.random() < cfg.missing_rate:
                    v0 = np.nan
                if rng.random() < cfg.missing_rate:
                    v24 = np.nan
            
            rows.append({
                'part_id': part_id,
                'lot_id': lot_id,
                'value_0h': v0,
                'value_24h': v24,
                'value_96h': v96,
                'value_168h': v168,
                'is_defect': 1 if is_defective else 0,
                'defect_type': defect_type,
                'defect_severity': round(severity, 4),
                'datasheet_limit': cfg.datasheet_limit,
                'lot_median_true': round(lot_median_base, 4),
                'heavy_tail_lot': 1 if is_heavy_tail else 0,
            })
    
    return pd.DataFrame(rows)


def split_lots(df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15,
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lot-grouped split ensuring zero data leakage."""
    rng = np.random.default_rng(random_state)
    lots = np.sort(df['lot_id'].unique())
    rng.shuffle(lots)
    
    n_lots = len(lots)
    n_train = int(n_lots * train_ratio)
    n_val = int(n_lots * val_ratio)
    
    train_lots = lots[:n_train]
    val_lots = lots[n_train:n_train + n_val]
    test_lots = lots[n_train + n_val:]
    
    return (
        df[df['lot_id'].isin(train_lots)].copy().reset_index(drop=True),
        df[df['lot_id'].isin(val_lots)].copy().reset_index(drop=True),
        df[df['lot_id'].isin(test_lots)].copy().reset_index(drop=True),
    )


def generate_stress_suite(output_dir: str = "data/stress") -> Dict[str, pd.DataFrame]:
    """Generates all stress-test scenario datasets."""
    os.makedirs(output_dir, exist_ok=True)
    results = {}
    
    for name, cfg in STRESS_CONFIGS.items():
        df = generate_physics_dataset(cfg)
        path = os.path.join(output_dir, f"stress_{name}.csv")
        df.to_csv(path, index=False)
        results[name] = df
        n_def = df['is_defect'].sum()
        print(f"  [{name}] {len(df)} parts, {df['lot_id'].nunique()} lots, "
              f"{n_def} defects ({n_def/len(df)*100:.1f}%)")
    
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Physics-based burn-in data generator")
    parser.add_argument('--n-lots', type=int, default=45)
    parser.add_argument('--output-dir', type=str, default='data/physics')
    parser.add_argument('--held-out', action='store_true', help='Use held-out configuration')
    parser.add_argument('--stress', action='store_true', help='Generate stress-test suite')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.stress:
        print("Generating stress-test suite...")
        generate_stress_suite(os.path.join(args.output_dir, 'stress'))
        print("Done.")
    else:
        cfg = HELD_OUT_CONFIG if args.held_out else DEFAULT_CONFIG
        if args.seed is not None:
            cfg.random_state = args.seed
        cfg.n_lots = args.n_lots
        
        label = "held-out" if args.held_out else "default"
        print(f"Generating {label} physics dataset with {cfg.n_lots} lots (seed={cfg.random_state})...")
        df = generate_physics_dataset(cfg)
        
        full_path = os.path.join(args.output_dir, "burnin_physics_full.csv")
        df.to_csv(full_path, index=False)
        print(f"Saved: {full_path} ({len(df)} parts, {df['is_defect'].sum()} defects)")
        
        train_df, val_df, test_df = split_lots(df)
        train_df.to_csv(os.path.join(args.output_dir, "burnin_train.csv"), index=False)
        val_df.to_csv(os.path.join(args.output_dir, "burnin_val.csv"), index=False)
        test_df.to_csv(os.path.join(args.output_dir, "burnin_test.csv"), index=False)
        
        print(f"  Train: {len(train_df)} parts ({train_df['is_defect'].sum()} defects)")
        print(f"  Val:   {len(val_df)} parts ({val_df['is_defect'].sum()} defects)")
        print(f"  Test:  {len(test_df)} parts ({test_df['is_defect'].sum()} defects)")
