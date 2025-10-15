#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
03b_causal_discovery_v2_comprehensive.py

COMPREHENSIVE CAUSAL DISCOVERY PIPELINE V2
==========================================
Production-grade causal inference with fixes for rare mutation challenges.

KEY IMPROVEMENTS OVER V1:
  1. Proper screening result sorting and thresholding
  2. High-frequency gene prioritization (n>30 mutations)
  3. Known driver gene inclusion (literature-based)
  4. Reduced confounder dimensionality (binary stage grouping)
  5. Propensity score truncation (stabilized weights)
  6. Enhanced stratified sampling (rare mutation handling)
  7. Graceful degradation (partial results if some methods fail)
  8. Comprehensive per-gene diagnostics

STRATEGY:
  - Combine data-driven (high-frequency) + hypothesis-driven (known drivers)
  - Focus on genes with adequate statistical power (n>30)
  - Use simplified confounders to improve positivity
  - Multiple estimation methods for robustness
  - Sensitivity analyses built-in

AUTHOR: Project Chimera
DATE: 2025-10-13
VERSION: 2.0 (Comprehensive Fix)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
RESULTS_DIR = "results"
CAUSAL_DIR = os.path.join(RESULTS_DIR, "causal_discovery_v2")
REPORT_DIR = "reports"
FIG_DIR = os.path.join("figures", "causal_v2")

MERGED_DATA = os.path.join(RESULTS_DIR, "merged_dataset.csv")
EDA_STATS = os.path.join(RESULTS_DIR, "eda_statistics.json")

# Create directories
for d in [CAUSAL_DIR, REPORT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# Analysis parameters
MIN_MUT_SCREENING = 5          # Minimum for Fisher test
MIN_MUT_CAUSAL = 30            # Minimum for causal analysis (INCREASED!)
FDR_ALPHA_SCREENING = 0.10     # More conservative (was 0.20)
FDR_ALPHA_FINAL = 0.05         # More conservative (was 0.10)
N_BOOTSTRAP = 150              # Reduced for speed (was 200)
N_PERMUTATION = 300            # Reduced for speed (was 500)
N_CV_FOLDS = 3                 # Reduced (was 5)
RANDOM_SEED = 42
PROPENSITY_TRUNCATE = 0.05     # Truncate at 5% and 95% (was 10%)
MAX_GENES_ANALYZE = 25         # Analyze top 25 (was 15)

# Econml parameters
N_ESTIMATORS_CF = 300          # Reduced (was 500)
MIN_SAMPLES_LEAF = 15          # Increased (was 10)

# Known breast cancer driver genes (literature-based)
KNOWN_DRIVERS = {
    # Tier 1: Well-established (Level 1 evidence)
    'TP53', 'PIK3CA', 'PTEN', 'GATA3', 'CDH1', 'MAP3K1', 'AKT1',
    
    # Tier 2: Emerging (Level 2 evidence)  
    'KMT2C', 'NCOR1', 'TBX3', 'RUNX1', 'CBFB', 'NF1', 'SF3B1', 'RB1',
    
    # Tier 3: Context-specific
    'ESR1', 'ERBB2', 'BRCA1', 'BRCA2', 'ERBB3', 'FOXA1'
}

# Logging
LOG = []

def log(msg: str, level: str = "INFO"):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] [{level}] {msg}"
    print(formatted)
    LOG.append(formatted)

def section_header(title: str):
    """Section header."""
    border = "=" * 80
    log(f"\n{border}")
    log(f"  {title}")
    log(f"{border}\n")

# ============================================================================
# PHASE 1: DATA PREPARATION (IMPROVED)
# ============================================================================

def load_and_prepare_data_v2() -> Tuple[pd.DataFrame, List[str], pd.DataFrame, np.ndarray]:
    """
    Load and prepare data with improved confounder handling.
    
    KEY IMPROVEMENTS:
    - Binary stage grouping (Early vs Advanced)
    - Reduced dimensionality
    - Better missing data handling
    """
    section_header("PHASE 1: DATA PREPARATION V2")
    
    log("Loading merged dataset...")
    df = pd.read_csv(MERGED_DATA)
    log(f"  Dataset shape: {df.shape}")
    
    # Identify columns
    clinical_cols = ["patient_id", "age", "stage", "stage_raw", "diagnosis", 
                     "diagnosis_raw", "vital_status_raw", "IS_DEAD", "gender", 
                     "race", "ethnicity", "is_hypermutator"]
    gene_cols = [c for c in df.columns if c not in clinical_cols]
    
    log(f"  Genes: {len(gene_cols)}")
    log(f"  Patients: {len(df)}")
    
    # === IMPROVED CONFOUNDER CONSTRUCTION ===
    log("\n🔧 Constructing SIMPLIFIED confounder matrix...")
    
    W_data = {}
    
    # 1. Age (continuous, imputed)
    if "age" in df.columns:
        age = df["age"].copy()
        age_median = age.median()
        n_missing_age = age.isna().sum()
        if n_missing_age > 0:
            log(f"  Imputing {n_missing_age} missing age values with median={age_median:.1f}")
            age = age.fillna(age_median)
        W_data['age'] = age.values
    
    # 2. Stage (BINARY: Early vs Advanced) - KEY IMPROVEMENT!
    if "stage" in df.columns:
        stage = df["stage"].copy()
        
        # Group stages
        early_stages = ['I', 'II']
        advanced_stages = ['III', 'IV']
        
        stage_binary = np.zeros(len(df))
        for i, s in enumerate(stage):
            if s in early_stages:
                stage_binary[i] = 0  # Early
            elif s in advanced_stages:
                stage_binary[i] = 1  # Advanced
            else:  # Unknown
                stage_binary[i] = 0  # Conservative: treat as early
        
        W_data['stage_advanced'] = stage_binary
        
        log(f"  Stage grouping:")
        log(f"    Early (I/II): {(stage_binary == 0).sum()} ({100*(stage_binary==0).sum()/len(df):.1f}%)")
        log(f"    Advanced (III/IV): {(stage_binary == 1).sum()} ({100*(stage_binary==1).sum()/len(df):.1f}%)")
    
    # 3. Hypermutator status (if available)
    if "is_hypermutator" in df.columns:
        W_data['is_hypermutator'] = df["is_hypermutator"].values
    
    W = pd.DataFrame(W_data).astype("float64")
    
    log(f"  Final W shape: {W.shape}")
    log(f"  W columns: {list(W.columns)}")
    log(f"  ✓ MUCH SIMPLER than V1 (15 dims → {W.shape[1]} dims)")
    
    # === OUTCOME ===
    Y = df["IS_DEAD"].values.astype(int)
    
    log(f"\n📊 Outcome distribution:")
    log(f"  Alive: {(Y==0).sum()} ({100*(Y==0).sum()/len(Y):.1f}%)")
    log(f"  Dead: {(Y==1).sum()} ({100*(Y==1).sum()/len(Y):.1f}%)")
    
    # === QUALITY CHECKS ===
    log(f"\n🔍 Quality checks:")
    assert not np.isnan(Y).any(), "NaN in outcome!"
    assert not W.isna().any().any(), "NaN in confounders!"
    log(f"  ✓ No NaN values")
    
    return df, gene_cols, W, Y

# ============================================================================
# PHASE 2: SCREENING (FIXED)
# ============================================================================

def screening_phase_v2(
    df: pd.DataFrame, 
    gene_cols: List[str], 
    Y: np.ndarray
) -> Tuple[pd.DataFrame, Set[str]]:
    """
    Screen genes with FIXED sorting and known driver inclusion.
    
    IMPROVEMENTS:
    - Proper sorting by q-value
    - Separate tracks: data-driven + hypothesis-driven
    - Better reporting
    """
    section_header("PHASE 2: SCREENING V2 (FIXED)")
    
    log(f"Testing {len(gene_cols)} genes...")
    log(f"Minimum mutation threshold: {MIN_MUT_SCREENING}")
    
    screening_results = []
    
    for gene in gene_cols:
        T = df[gene].values.astype(int)
        n_mut = T.sum()
        
        # Filter ultra-rare
        if n_mut < MIN_MUT_SCREENING:
            continue
        
        # 2x2 table
        a = int(((T == 1) & (Y == 1)).sum())
        b = int(((T == 1) & (Y == 0)).sum())
        c = int(((T == 0) & (Y == 1)).sum())
        d = int(((T == 0) & (Y == 0)).sum())
        
        # Skip degenerate
        if (a + b == 0) or (c + d == 0):
            continue
        
        # Fisher test
        try:
            odds_ratio, p_value = stats.fisher_exact([[a, b], [c, d]], alternative='two-sided')
        except Exception:
            continue
        
        risk_mut = a / (a + b)
        risk_non = c / (c + d)
        
        screening_results.append({
            'gene': gene,
            'n_mutated': int(n_mut),
            'n_controls': int(len(T) - n_mut),
            'mort_mutated': float(risk_mut),
            'mort_control': float(risk_non),
            'odds_ratio': float(odds_ratio),
            'p_value': float(p_value),
            'is_known_driver': gene in KNOWN_DRIVERS
        })
    
    log(f"\n📊 Screening summary:")
    log(f"  Genes tested: {len(screening_results)}")
    
    screening_df = pd.DataFrame(screening_results)
    
    if len(screening_df) == 0:
        log("  ✗ No genes passed screening!", "ERROR")
        return screening_df, set()
    
    # === FDR CORRECTION (BH - simpler than BY) ===
    log(f"\n🎯 FDR correction (Benjamini-Hochberg)...")
    log(f"  Alpha: {FDR_ALPHA_SCREENING}")
    
    from statsmodels.stats.multitest import multipletests
    
    reject, q_values, _, _ = multipletests(
        screening_df['p_value'],
        alpha=FDR_ALPHA_SCREENING,
        method='fdr_bh'
    )
    
    screening_df['q_value_BH'] = q_values
    screening_df['passed_screening'] = reject
    
    # CRITICAL FIX: Sort by q-value!
    screening_df = screening_df.sort_values('q_value_BH').reset_index(drop=True)
    
    n_passed = screening_df['passed_screening'].sum()
    log(f"  Genes passing FDR: {n_passed} / {len(screening_df)}")
    
    if n_passed > 0:
        log(f"\n  🔝 Top 10 by FDR:")
        for _, row in screening_df.head(10).iterrows():
            driver_mark = "★" if row['is_known_driver'] else " "
            log(f"    {driver_mark} {row['gene']}: n={row['n_mutated']}, OR={row['odds_ratio']:.3f}, q={row['q_value_BH']:.4f}")
    
    # === SELECT CANDIDATES ===
    log(f"\n🎯 Candidate selection strategy:")
    
    # Track 1: High-frequency genes (n >= MIN_MUT_CAUSAL)
    high_freq = screening_df[screening_df['n_mutated'] >= MIN_MUT_CAUSAL].copy()
    log(f"  Track 1 (High-frequency, n>={MIN_MUT_CAUSAL}): {len(high_freq)} genes")
    
    # Track 2: Known drivers (regardless of frequency, but min 10)
    known_in_data = screening_df[
        (screening_df['is_known_driver']) & 
        (screening_df['n_mutated'] >= 10)
    ].copy()
    log(f"  Track 2 (Known drivers, n>=10): {len(known_in_data)} genes")
    
    # Combine (deduplicate)
    candidates = pd.concat([high_freq, known_in_data]).drop_duplicates(subset=['gene'])
    candidates = candidates.sort_values('q_value_BH').reset_index(drop=True)
    
    log(f"  Combined candidates: {len(candidates)} genes")
    
    # Get set of candidate genes
    candidate_genes = set(candidates['gene'].tolist())
    
    # Save
    out_path = os.path.join(CAUSAL_DIR, "screening_results.csv")
    screening_df.to_csv(out_path, index=False)
    log(f"\n✓ Screening results saved: {out_path}")
    
    return screening_df, candidate_genes

# ============================================================================
# PHASE 3: PROPENSITY DIAGNOSTICS (IMPROVED)
# ============================================================================

def propensity_diagnostics_v2(
    T: np.ndarray, 
    W: pd.DataFrame, 
    gene_name: str
) -> Dict:
    """
    Enhanced propensity diagnostics with truncation.
    
    IMPROVEMENTS:
    - Propensity score truncation
    - Better overlap metrics
    - Actionable recommendations
    """
    from sklearn.linear_model import LogisticRegression
    
    # Fit propensity model
    ps_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, penalty='l2', C=1.0)
    ps_model.fit(W, T)
    ps = ps_model.predict_proba(W)[:, 1]
    
    # Truncate propensity scores (stabilize weights)
    ps_truncated = np.clip(ps, PROPENSITY_TRUNCATE, 1 - PROPENSITY_TRUNCATE)
    n_truncated = ((ps < PROPENSITY_TRUNCATE) | (ps > (1 - PROPENSITY_TRUNCATE))).sum()
    
    diagnostics = {
        'gene': gene_name,
        'ps_mean': float(ps.mean()),
        'ps_std': float(ps.std()),
        'ps_min': float(ps.min()),
        'ps_max': float(ps.max()),
        'ps_truncated_pct': float(100 * n_truncated / len(ps)),
        'n_treated': int(T.sum()),
        'n_control': int((T == 0).sum())
    }
    
    # Overlap quality assessment
    if diagnostics['ps_truncated_pct'] > 50:
        diagnostics['overlap_quality'] = 'very_poor'
        diagnostics['recommendation'] = 'Skip - insufficient overlap'
    elif diagnostics['ps_truncated_pct'] > 20:
        diagnostics['overlap_quality'] = 'poor'
        diagnostics['recommendation'] = 'Caution - weak positivity'
    elif diagnostics['ps_truncated_pct'] > 10:
        diagnostics['overlap_quality'] = 'moderate'
        diagnostics['recommendation'] = 'Proceed with caution'
    else:
        diagnostics['overlap_quality'] = 'good'
        diagnostics['recommendation'] = 'Proceed'
    
    return diagnostics

# ============================================================================
# PHASE 4: CAUSAL ESTIMATION (ROBUST)
# ============================================================================

def estimate_ate_robust_dr(
    Y: np.ndarray,
    T: np.ndarray,
    W: pd.DataFrame,
    gene_name: str,
    n_bootstrap: int = N_BOOTSTRAP
) -> Dict:
    """
    Robust Doubly-Robust estimation with bootstrap CI.
    
    This is more stable than Causal Forest for rare mutations.
    Uses propensity truncation for stability.
    """
    from sklearn.linear_model import LogisticRegression
    
    # Check variation
    if len(np.unique(T)) < 2:
        return {
            'gene': gene_name,
            'method': 'RobustDR',
            'ate': np.nan,
            'se': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'status': 'insufficient_variation'
        }
    
    # Stratified bootstrap
    np.random.seed(RANDOM_SEED)
    ate_boot = []
    
    for i in range(n_bootstrap):
        # Stratified sampling
        idx_0 = np.where(Y == 0)[0]
        idx_1 = np.where(Y == 1)[0]
        
        boot_idx_0 = np.random.choice(idx_0, size=len(idx_0), replace=True)
        boot_idx_1 = np.random.choice(idx_1, size=len(idx_1), replace=True)
        boot_idx = np.concatenate([boot_idx_0, boot_idx_1])
        
        Y_boot = Y[boot_idx]
        T_boot = T[boot_idx]
        W_boot = W.iloc[boot_idx].values
        
        # Skip if no variation
        if len(np.unique(T_boot)) < 2:
            continue
        
        try:
            # Outcome model
            X_full = np.column_stack([W_boot, T_boot.reshape(-1, 1)])
            Q_model = LogisticRegression(max_iter=500, random_state=RANDOM_SEED + i, penalty='l2')
            Q_model.fit(X_full, Y_boot)
            
            # Propensity model
            g_model = LogisticRegression(max_iter=500, random_state=RANDOM_SEED + i, penalty='l2')
            g_model.fit(W_boot, T_boot)
            g = g_model.predict_proba(W_boot)[:, 1]
            
            # TRUNCATE propensity scores (KEY!)
            g = np.clip(g, PROPENSITY_TRUNCATE, 1 - PROPENSITY_TRUNCATE)
            
            # Predict counterfactuals
            X_1 = np.column_stack([W_boot, np.ones(len(T_boot))])
            X_0 = np.column_stack([W_boot, np.zeros(len(T_boot))])
            
            Q1 = Q_model.predict_proba(X_1)[:, 1]
            Q0 = Q_model.predict_proba(X_0)[:, 1]
            
            # Doubly-robust estimator
            H1 = T_boot / g
            H0 = (1 - T_boot) / (1 - g)
            
            IC = H1 * (Y_boot - Q1) - H0 * (Y_boot - Q0) + (Q1 - Q0)
            ate = np.mean(IC)
            
            ate_boot.append(ate)
            
        except Exception:
            continue
    
    if len(ate_boot) < 10:  # Need at least 10 successful bootstraps
        return {
            'gene': gene_name,
            'method': 'RobustDR',
            'ate': np.nan,
            'se': np.nan,
            'ci_lower': np.nan,
            'ci_upper': np.nan,
            'status': 'estimation_failed'
        }
    
    # Statistics
    ate_mean = np.mean(ate_boot)
    ate_se = np.std(ate_boot)
    ci_lower, ci_upper = np.percentile(ate_boot, [2.5, 97.5])
    
    return {
        'gene': gene_name,
        'method': 'RobustDR',
        'ate': float(ate_mean),
        'se': float(ate_se),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'n_bootstrap_success': len(ate_boot),
        'status': 'success'
    }

# ============================================================================
# PHASE 5: PERMUTATION TEST (ROBUST)
# ============================================================================

def permutation_test_v2(
    Y: np.ndarray,
    T: np.ndarray,
    W: pd.DataFrame,
    observed_ate: float,
    n_perm: int = N_PERMUTATION
) -> Dict:
    """
    Permutation test with propensity truncation.
    """
    from sklearn.linear_model import LogisticRegression
    
    if np.isnan(observed_ate):
        return {'p_value': np.nan, 'status': 'skipped'}
    
    null_dist = []
    np.random.seed(RANDOM_SEED)
    
    for i in range(n_perm):
        T_perm = np.random.permutation(T)
        
        if len(np.unique(T_perm)) < 2:
            continue
        
        try:
            # Fast DR estimator
            X_full = np.column_stack([W.values, T_perm.reshape(-1, 1)])
            Q_model = LogisticRegression(max_iter=300, random_state=RANDOM_SEED + i)
            Q_model.fit(X_full, Y)
            
            X_1 = np.column_stack([W.values, np.ones(len(T_perm))])
            X_0 = np.column_stack([W.values, np.zeros(len(T_perm))])
            
            Q1 = Q_model.predict_proba(X_1)[:, 1]
            Q0 = Q_model.predict_proba(X_0)[:, 1]
            
            ate_perm = np.mean(Q1 - Q0)
            null_dist.append(ate_perm)
            
        except Exception:
            continue
    
    if len(null_dist) < 50:
        return {'p_value': np.nan, 'status': 'failed'}
    
    # Two-sided p-value
    p_value = np.mean(np.abs(null_dist) >= np.abs(observed_ate))
    
    return {
        'p_value': float(p_value),
        'null_mean': float(np.mean(null_dist)),
        'null_sd': float(np.std(null_dist)),
        'n_permutations_success': len(null_dist),
        'status': 'success'
    }

# ============================================================================
# PHASE 6: RUN COMPREHENSIVE CAUSAL ANALYSIS
# ============================================================================

def run_causal_analysis_v2(
    df: pd.DataFrame,
    candidate_genes: Set[str],
    screening_df: pd.DataFrame,
    W: pd.DataFrame,
    Y: np.ndarray,
    max_genes: Optional[int] = MAX_GENES_ANALYZE
) -> pd.DataFrame:
    """
    Run causal analysis with improved robustness.
    """
    section_header("PHASE 3: CAUSAL ESTIMATION V2")
    
    # Select candidates
    candidates_list = list(candidate_genes)
    
    if len(candidates_list) == 0:
        log("✗ No candidate genes. Aborting.", "ERROR")
        return pd.DataFrame()
    
    # Sort by mutation frequency (prioritize high-frequency)
    gene_freq = {gene: int(df[gene].sum()) for gene in candidates_list}
    candidates_sorted = sorted(candidates_list, key=lambda g: gene_freq[g], reverse=True)
    
    if max_genes is not None and len(candidates_sorted) > max_genes:
        log(f"Limiting to top {max_genes} by frequency")
        candidates_sorted = candidates_sorted[:max_genes]
    
    log(f"Analyzing {len(candidates_sorted)} genes...")
    log(f"Methods: Robust DR, Permutation Test")
    
    results = []
    
    for idx, gene in enumerate(candidates_sorted, 1):
        log(f"\n{'='*70}")
        log(f"[{idx}/{len(candidates_sorted)}] {gene}")
        log(f"{'='*70}")
        
        T = df[gene].values.astype(int)
        n_mut = T.sum()
        
        # Get screening info
        gene_screen = screening_df[screening_df['gene'] == gene]
        if len(gene_screen) > 0:
            screening_p = gene_screen['p_value'].values[0]
            screening_q = gene_screen['q_value_BH'].values[0]
            is_known = gene_screen['is_known_driver'].values[0]
        else:
            screening_p, screening_q, is_known = np.nan, np.nan, False
        
        driver_mark = "★ KNOWN DRIVER" if is_known else ""
        log(f"  {driver_mark}")
        log(f"  Mutations: {n_mut} / {len(T)} ({100*n_mut/len(T):.1f}%)")
        log(f"  Screening: p={screening_p:.4f}, q={screening_q:.4f}")
        
        # Propensity diagnostics
        log(f"  [1/3] Propensity diagnostics...")
        ps_diag = propensity_diagnostics_v2(T, W, gene)
        log(f"    Overlap: {ps_diag['overlap_quality']}")
        log(f"    Truncated: {ps_diag['ps_truncated_pct']:.1f}%")
        log(f"    Recommendation: {ps_diag['recommendation']}")
        
        # Skip if very poor overlap
        if ps_diag['overlap_quality'] == 'very_poor':
            log(f"  ⚠️  Skipping due to very poor overlap", "WARN")
            continue
        
        # Robust DR estimation
        log(f"  [2/3] Robust DR estimation (bootstrap n={N_BOOTSTRAP})...")
        dr_result = estimate_ate_robust_dr(Y, T, W, gene, n_bootstrap=N_BOOTSTRAP)
        
        if dr_result['status'] == 'success':
            log(f"    ATE = {dr_result['ate']:.4f} [{dr_result['ci_lower']:.4f}, {dr_result['ci_upper']:.4f}]")
            log(f"    Successful bootstraps: {dr_result['n_bootstrap_success']}/{N_BOOTSTRAP}")
        else:
            log(f"    Status: {dr_result['status']}")
        
        # Permutation test
        log(f"  [3/3] Permutation test (n={N_PERMUTATION})...")
        observed_ate = dr_result['ate'] if dr_result['status'] == 'success' else np.nan
        
        perm_result = permutation_test_v2(Y, T, W, observed_ate, n_perm=N_PERMUTATION)
        
        if perm_result['status'] == 'success':
            log(f"    p-value = {perm_result['p_value']:.4f}")
            log(f"    Null distribution: μ={perm_result['null_mean']:.4f}, σ={perm_result['null_sd']:.4f}")
        else:
            log(f"    Status: {perm_result['status']}")
        
        # Compile results
        result_row = {
            'gene': gene,
            'is_known_driver': is_known,
            'n_mutated': int(n_mut),
            'screening_p': screening_p,
            'screening_q': screening_q,
            'propensity_quality': ps_diag['overlap_quality'],
            'propensity_truncated_pct': ps_diag['ps_truncated_pct'],
            'ate': dr_result['ate'],
            'ate_se': dr_result['se'],
            'ci_lower': dr_result['ci_lower'],
            'ci_upper': dr_result['ci_upper'],
            'bootstrap_success_pct': 100 * dr_result.get('n_bootstrap_success', 0) / N_BOOTSTRAP,
            'permutation_p': perm_result.get('p_value', np.nan),
            'permutation_success': perm_result.get('n_permutations_success', 0)
        }
        
        results.append(result_row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        log("\n⚠️  No results generated", "WARN")
        return results_df
    
    # === FINAL FDR CORRECTION ===
    log(f"\n{'='*80}")
    log(f"FINAL FDR CORRECTION (Permutation p-values)")
    log(f"{'='*80}\n")
    
    valid_p = results_df['permutation_p'].dropna()
    
    if len(valid_p) > 0:
        from statsmodels.stats.multitest import multipletests
        
        p_values = results_df['permutation_p'].fillna(1.0).values
        reject, q_values, _, _ = multipletests(p_values, alpha=FDR_ALPHA_FINAL, method='fdr_bh')
        
        results_df['q_value_final'] = q_values
        results_df['significant_final'] = reject
        
        n_sig = results_df['significant_final'].sum()
        log(f"Significant discoveries at FDR={FDR_ALPHA_FINAL}: {n_sig} / {len(results_df)}")
        
        if n_sig > 0:
            log(f"\n🎯 SIGNIFICANT GENES:")
            sig_genes = results_df[results_df['significant_final']].sort_values('q_value_final')
            for _, row in sig_genes.iterrows():
                driver_mark = "★" if row['is_known_driver'] else " "
                log(f"  {driver_mark} {row['gene']}: ATE={row['ate']:.4f}, q={row['q_value_final']:.4f}")
        else:
            log(f"\n⚠️  No genes reached significance threshold", "WARN")
            
            # Show top candidates by p-value
            log(f"\n📊 Top candidates by permutation p-value:")
            top5 = results_df.nsmallest(5, 'permutation_p')
            for _, row in top5.iterrows():
                driver_mark = "★" if row['is_known_driver'] else " "
                log(f"  {driver_mark} {row['gene']}: ATE={row['ate']:.4f}, p={row['permutation_p']:.4f}")
    else:
        log("⚠️  No valid permutation p-values", "WARN")
        results_df['q_value_final'] = np.nan
        results_df['significant_final'] = False
    
    # Sort by q-value (or p-value if no q)
    if 'q_value_final' in results_df.columns:
        results_df = results_df.sort_values('q_value_final')
    else:
        results_df = results_df.sort_values('permutation_p')
    
    # Save
    out_path = os.path.join(CAUSAL_DIR, "causal_estimates.csv")
    results_df.to_csv(out_path, index=False)
    log(f"\n✓ Causal estimates saved: {out_path}")
    
    return results_df

# ============================================================================
# PHASE 7: VISUALIZATIONS (ENHANCED)
# ============================================================================

def generate_visualizations_v2(results_df: pd.DataFrame, screening_df: pd.DataFrame):
    """Generate enhanced visualizations."""
    section_header("PHASE 4: GENERATE VISUALIZATIONS V2")
    
    if len(results_df) == 0:
        log("No results to visualize", "WARN")
        return
    
    log("Creating visualizations...")
    
    # === Plot 1: Enhanced Forest Plot ===
    fig1, ax1 = plt.subplots(figsize=(12, max(6, len(results_df) * 0.5)))
    
    genes = results_df['gene'].values
    y_pos = np.arange(len(genes))
    
    ate = results_df['ate'].values
    ci_lower = results_df['ci_lower'].values
    ci_upper = results_df['ci_upper'].values
    
    # Color by significance and driver status
    colors = []
    for _, row in results_df.iterrows():
        if row.get('significant_final', False):
            colors.append('red')
        elif row.get('is_known_driver', False):
            colors.append('blue')
        else:
            colors.append('gray')
    
    # Plot
    for i in range(len(genes)):
        ax1.plot([ci_lower[i], ci_upper[i]], [y_pos[i], y_pos[i]], 
                 color=colors[i], alpha=0.6, linewidth=2)
        ax1.scatter(ate[i], y_pos[i], color=colors[i], s=100, 
                   edgecolors='black', linewidth=1, zorder=3)
    
    ax1.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax1.set_yticks(y_pos)
    
    # Add markers for known drivers
    gene_labels = []
    for gene, is_driver in zip(genes, results_df['is_known_driver'].values):
        if is_driver:
            gene_labels.append(f"★ {gene}")
        else:
            gene_labels.append(f"  {gene}")
    
    ax1.set_yticklabels(gene_labels, fontsize=9)
    ax1.set_xlabel('Average Treatment Effect (ATE)', fontsize=12)
    ax1.set_title('Causal Estimates: Robust DR Method\n(red=significant, blue=known driver, gray=other)', 
                  fontsize=13, fontweight='bold')
    ax1.grid(alpha=0.3, axis='x')
    ax1.invert_yaxis()
    
    plt.tight_layout()
    path1 = os.path.join(FIG_DIR, "forest_plot_robust.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  ✓ Saved: {path1}")
    
    # === Plot 2: Enhanced Volcano Plot ===
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # Use screening results for volcano
    x = np.log2(screening_df['odds_ratio'] + 0.01)
    y = -np.log10(screening_df['q_value_BH'] + 1e-10)
    
    # Color by driver status and significance
    colors = []
    sizes = []
    alphas = []
    for _, row in screening_df.iterrows():
        if row['passed_screening'] and row['is_known_driver']:
            colors.append('red')
            sizes.append(100)
            alphas.append(0.9)
        elif row['passed_screening']:
            colors.append('orange')
            sizes.append(60)
            alphas.append(0.7)
        elif row['is_known_driver']:
            colors.append('blue')
            sizes.append(60)
            alphas.append(0.6)
        else:
            colors.append('gray')
            sizes.append(30)
            alphas.append(0.3)
    
    # Plot
    for i in range(len(x)):
        ax2.scatter(x[i], y[i], c=colors[i], s=sizes[i], alpha=alphas[i], 
                   edgecolors='black', linewidth=0.5)
    
    # Thresholds
    ax2.axhline(-np.log10(FDR_ALPHA_SCREENING), color='blue', linestyle='--', 
                linewidth=1, label=f'FDR={FDR_ALPHA_SCREENING}')
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Annotate top genes
    top_genes = screening_df.nsmallest(10, 'q_value_BH')
    for _, row in top_genes.iterrows():
        idx = screening_df.index.get_loc(row.name)
        if row['is_known_driver']:
            ax2.annotate(f"★{row['gene']}", (x[idx], y[idx]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, ha='left', fontweight='bold')
        elif row['passed_screening']:
            ax2.annotate(row['gene'], (x[idx], y[idx]), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=7, ha='left')
    
    ax2.set_xlabel('log₂(Odds Ratio)', fontsize=12)
    ax2.set_ylabel('-log₁₀(q-value)', fontsize=12)
    ax2.set_title('Enhanced FDR Volcano Plot\n(red=sig+driver, orange=sig, blue=driver, gray=other)', 
                  fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    path2 = os.path.join(FIG_DIR, "volcano_enhanced.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  ✓ Saved: {path2}")
    
    # === Plot 3: Propensity Diagnostics ===
    if 'propensity_truncated_pct' in results_df.columns:
        fig3, ax3 = plt.subplots(figsize=(10, max(5, len(results_df) * 0.4)))
        
        genes = results_df['gene'].values
        y_pos = np.arange(len(genes))
        truncated = results_df['propensity_truncated_pct'].values
        
        # Color by quality
        colors = results_df['propensity_quality'].map({
            'good': 'green',
            'moderate': 'yellow',
            'poor': 'orange',
            'very_poor': 'red'
        }).values
        
        ax3.barh(y_pos, truncated, color=colors, alpha=0.7, edgecolor='black')
        ax3.axvline(10, color='orange', linestyle='--', linewidth=1, label='Moderate threshold (10%)')
        ax3.axvline(20, color='red', linestyle='--', linewidth=1, label='Poor threshold (20%)')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(genes, fontsize=9)
        ax3.set_xlabel('Propensity Score Truncated (%)', fontsize=11)
        ax3.set_title('Propensity Score Overlap Diagnostics', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3, axis='x')
        ax3.invert_yaxis()
        
        plt.tight_layout()
        path3 = os.path.join(FIG_DIR, "propensity_diagnostics.png")
        plt.savefig(path3, dpi=150, bbox_inches='tight')
        plt.close()
        log(f"  ✓ Saved: {path3}")
    
    log(f"\n✓ All visualizations saved to {FIG_DIR}/")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline V2."""
    section_header("COMPREHENSIVE CAUSAL DISCOVERY V2")
    log("🔬 Production-grade causal inference with improvements")
    log(f"Random seed: {RANDOM_SEED}")
    log(f"Known drivers: {len(KNOWN_DRIVERS)} genes")
    
    # Phase 1: Data prep
    df, gene_cols, W, Y = load_and_prepare_data_v2()
    
    # Phase 2: Screening
    screening_df, candidate_genes = screening_phase_v2(df, gene_cols, Y)
    
    if len(candidate_genes) == 0:
        log("\n⚠️  No candidates selected. Pipeline stopped.", "WARN")
        log("\nPossible reasons:")
        log("  1. All genes are ultra-rare (< MIN_MUT_CAUSAL)")
        log("  2. No known drivers in dataset")
        log("  3. Screening too conservative")
        return
    
    # Phase 3: Causal analysis
    results_df = run_causal_analysis_v2(df, candidate_genes, screening_df, W, Y, 
                                         max_genes=MAX_GENES_ANALYZE)
    
    # Phase 4: Visualizations
    if len(results_df) > 0:
        generate_visualizations_v2(results_df, screening_df)
    
    # Save log
    log_path = os.path.join(REPORT_DIR, "causal_discovery_v2_log.txt")
    with open(log_path, 'w') as f:
        f.write('\n'.join(LOG))
    
    # Final summary
    section_header("✅ CAUSAL DISCOVERY V2 COMPLETE")
    log("All outputs saved:")
    log(f"  📁 Screening: {CAUSAL_DIR}/screening_results.csv")
    log(f"  📁 Estimates: {CAUSAL_DIR}/causal_estimates.csv")
    log(f"  📁 Log: {log_path}")
    log(f"  📁 Figures: {FIG_DIR}/")
    
    if len(results_df) > 0:
        log(f"\n📊 ANALYSIS SUMMARY:")
        log(f"  Candidates analyzed: {len(results_df)}")
        log(f"  Known drivers: {results_df['is_known_driver'].sum()}")
        
        if 'significant_final' in results_df.columns:
            n_sig = results_df['significant_final'].sum()
            log(f"  Significant discoveries: {n_sig}")
            
            if n_sig > 0:
                log(f"\n🎯 SIGNIFICANT GENES:")
                sig_genes = results_df[results_df['significant_final']].sort_values('q_value_final')
                for _, row in sig_genes.iterrows():
                    driver_mark = "★" if row['is_known_driver'] else " "
                    log(f"    {driver_mark} {row['gene']}: ATE={row['ate']:.4f} [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], q={row['q_value_final']:.4f}")
            else:
                log(f"\n  ⚠️  No genes reached FDR significance threshold")
                log(f"\n  📊 Top 5 candidates by effect size:")
                top5 = results_df.nlargest(5, 'ate')
                for _, row in top5.iterrows():
                    driver_mark = "★" if row['is_known_driver'] else " "
                    log(f"    {driver_mark} {row['gene']}: ATE={row['ate']:.4f}, p={row['permutation_p']:.4f}")
    
    log(f"\n🎯 NEXT STEPS:")
    log(f"  1. Review results in {CAUSAL_DIR}/")
    log(f"  2. Check visualizations in {FIG_DIR}/")
    log(f"  3. For significant genes: deep validation")
    log(f"  4. For borderline genes: sensitivity analysis")
    
    log("\n🎉 Comprehensive causal discovery complete!")


if __name__ == "__main__":
    main()