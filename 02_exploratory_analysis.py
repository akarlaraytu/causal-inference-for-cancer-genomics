#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_exploratory_analysis.py

EXPLORATORY DATA ANALYSIS (EDA)
================================
Foundation for causal discovery - understand the data before causal modeling.

OBJECTIVES:
  1. Clinical features vs outcome analysis
  2. Top mutated genes characterization
  3. Hypermutator deep dive
  4. Confounding structure exploration
  5. Survival analysis (Kaplan-Meier)

INPUTS:
  - results/merged_dataset.csv

OUTPUTS:
  - reports/eda_summary.txt
  - figures/eda/ (multiple plots)
  - results/eda_statistics.json

This is NOT hypothesis testing - just exploration to inform causal analysis.
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

warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = "results"
REPORT_DIR = "reports"
FIG_DIR = os.path.join("figures", "eda")

MERGED_DATA = os.path.join(RESULTS_DIR, "merged_dataset.csv")
EDA_REPORT = os.path.join(REPORT_DIR, "eda_summary.txt")
EDA_STATS = os.path.join(RESULTS_DIR, "eda_statistics.json")

# Create directories
for d in [REPORT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# Logging
LOG = []

def log(msg, level="INFO"):
    """Log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] [{level}] {msg}"
    print(formatted)
    LOG.append(formatted)

def section_header(title):
    """Section header."""
    border = "=" * 80
    log(f"\n{border}")
    log(f"  {title}")
    log(f"{border}\n")

# ============================================================================
# PHASE 1: LOAD AND PREP DATA
# ============================================================================

def load_data():
    """Load merged dataset."""
    section_header("PHASE 1: LOAD DATA")
    
    log(f"Loading merged dataset: {MERGED_DATA}")
    df = pd.read_csv(MERGED_DATA)
    
    log(f"  Shape: {df.shape}")
    log(f"  Patients: {len(df)}")
    log(f"  Features: {df.shape[1]}")
    
    # Identify gene columns
    clinical_cols = ["patient_id", "age", "stage", "stage_raw", "diagnosis", 
                     "diagnosis_raw", "vital_status_raw", "IS_DEAD", "gender", 
                     "race", "ethnicity", "is_hypermutator"]
    gene_cols = [c for c in df.columns if c not in clinical_cols]
    
    log(f"  Clinical features: {len([c for c in clinical_cols if c in df.columns])}")
    log(f"  Gene features: {len(gene_cols)}")
    
    # Basic outcome check
    outcome_dist = df["IS_DEAD"].value_counts()
    log(f"\n  Outcome distribution:")
    for outcome, count in outcome_dist.items():
        label = "Dead" if outcome == 1 else "Alive"
        pct = 100 * count / len(df)
        log(f"    {label}: {count} ({pct:.1f}%)")
    
    return df, gene_cols

# ============================================================================
# PHASE 2: CLINICAL FEATURES VS OUTCOME
# ============================================================================

def analyze_clinical_vs_outcome(df):
    """Analyze clinical features vs outcome."""
    section_header("PHASE 2: CLINICAL FEATURES VS OUTCOME")
    
    stats_results = {}
    
    # === AGE vs OUTCOME ===
    log("📅 AGE vs OUTCOME")
    
    age_alive = df.loc[df["IS_DEAD"] == 0, "age"].dropna()
    age_dead = df.loc[df["IS_DEAD"] == 1, "age"].dropna()
    
    log(f"  Alive: mean={age_alive.mean():.1f}, median={age_alive.median():.1f}")
    log(f"  Dead:  mean={age_dead.mean():.1f}, median={age_dead.median():.1f}")
    
    # T-test
    t_stat, p_val = stats.ttest_ind(age_alive, age_dead)
    log(f"  T-test: t={t_stat:.3f}, p={p_val:.4f}")
    
    if p_val < 0.05:
        log(f"  ✓ Significant difference (p<0.05)")
    else:
        log(f"  ✗ No significant difference")
    
    stats_results['age_vs_outcome'] = {
        'alive_mean': float(age_alive.mean()),
        'dead_mean': float(age_dead.mean()),
        't_statistic': float(t_stat),
        'p_value': float(p_val)
    }
    
    # === STAGE vs OUTCOME ===
    log("\n🎚️  STAGE vs OUTCOME")
    
    stage_outcome = pd.crosstab(df["stage"], df["IS_DEAD"], normalize='index') * 100
    log(f"  Mortality rate by stage:")
    for stage in stage_outcome.index:
        if 1 in stage_outcome.columns:
            mortality = stage_outcome.loc[stage, 1]
            log(f"    {stage}: {mortality:.1f}%")
    
    # Chi-square test
    contingency = pd.crosstab(df["stage"], df["IS_DEAD"])
    chi2, p_val, dof, expected = stats.chi2_contingency(contingency)
    log(f"  Chi-square: χ²={chi2:.3f}, p={p_val:.4f}, dof={dof}")
    
    if p_val < 0.05:
        log(f"  ✓ Significant association (p<0.05)")
    else:
        log(f"  ✗ No significant association")
    
    stats_results['stage_vs_outcome'] = {
        'chi2_statistic': float(chi2),
        'p_value': float(p_val),
        'degrees_of_freedom': int(dof)
    }
    
    # === DIAGNOSIS vs OUTCOME ===
    log("\n🏥 DIAGNOSIS vs OUTCOME (top types)")
    
    top_diagnoses = df["diagnosis"].value_counts().head(5).index
    diag_outcome = df[df["diagnosis"].isin(top_diagnoses)].groupby("diagnosis")["IS_DEAD"].agg(['sum', 'count'])
    diag_outcome['mortality_rate'] = 100 * diag_outcome['sum'] / diag_outcome['count']
    
    log(f"  Mortality rate by diagnosis:")
    for diagnosis, row in diag_outcome.iterrows():
        log(f"    {diagnosis[:50]}: {row['mortality_rate']:.1f}% ({int(row['sum'])}/{int(row['count'])})")
    
    stats_results['diagnosis_types'] = diag_outcome['mortality_rate'].to_dict()
    
    return stats_results

# ============================================================================
# PHASE 3: TOP MUTATED GENES ANALYSIS
# ============================================================================

def analyze_top_genes(df, gene_cols, top_n=20):
    """Analyze top mutated genes."""
    section_header(f"PHASE 3: TOP {top_n} MUTATED GENES ANALYSIS")
    
    # Gene mutation frequencies
    gene_sums = df[gene_cols].sum().sort_values(ascending=False)
    top_genes = gene_sums.head(top_n)
    
    log(f"🔝 Top {top_n} mutated genes:")
    for i, (gene, count) in enumerate(top_genes.items(), 1):
        pct = 100 * count / len(df)
        log(f"  {i:2d}. {gene}: {int(count)} ({pct:.1f}%)")
    
    # === Analyze association with outcome ===
    log(f"\n🎯 Association with outcome (Fisher exact test):")
    
    gene_outcome_stats = []
    
    for gene in top_genes.index[:10]:  # Test top 10
        # 2x2 contingency table
        mut = df[gene]
        dead = df["IS_DEAD"]
        
        a = int(((mut == 1) & (dead == 1)).sum())  # mut & dead
        b = int(((mut == 1) & (dead == 0)).sum())  # mut & alive
        c = int(((mut == 0) & (dead == 1)).sum())  # non-mut & dead
        d = int(((mut == 0) & (dead == 0)).sum())  # non-mut & alive
        
        # Fisher exact test
        odds_ratio, p_val = stats.fisher_exact([[a, b], [c, d]], alternative='two-sided')
        
        # Mortality rates
        mort_mut = 100 * a / max(a + b, 1)
        mort_non = 100 * c / max(c + d, 1)
        
        gene_outcome_stats.append({
            'gene': gene,
            'mut_count': int(a + b),
            'mortality_mut': mort_mut,
            'mortality_non': mort_non,
            'odds_ratio': odds_ratio,
            'p_value': p_val
        })
        
        log(f"  {gene}:")
        log(f"    Mortality (mutated): {mort_mut:.1f}% ({a}/{a+b})")
        log(f"    Mortality (non-mutated): {mort_non:.1f}% ({c}/{c+d})")
        log(f"    Odds ratio: {odds_ratio:.3f}, p={p_val:.4f}")
    
    return gene_outcome_stats, top_genes

# ============================================================================
# PHASE 4: HYPERMUTATOR ANALYSIS
# ============================================================================

def analyze_hypermutators(df, gene_cols):
    """Deep dive into hypermutators."""
    section_header("PHASE 4: HYPERMUTATOR DEEP DIVE")
    
    hyper_patients = df[df["is_hypermutator"] == 1]
    normal_patients = df[df["is_hypermutator"] == 0]
    
    log(f"📊 Hypermutator vs Normal comparison:")
    log(f"  Hypermutators: {len(hyper_patients)}")
    log(f"  Normal: {len(normal_patients)}")
    
    # Mutation burden comparison
    hyper_burden = hyper_patients[gene_cols].sum(axis=1)
    normal_burden = normal_patients[gene_cols].sum(axis=1)
    
    log(f"\n  Mutation burden:")
    log(f"    Hypermutators: mean={hyper_burden.mean():.0f}, median={hyper_burden.median():.0f}")
    log(f"    Normal: mean={normal_burden.mean():.0f}, median={normal_burden.median():.0f}")
    log(f"    Fold difference: {hyper_burden.mean() / normal_burden.mean():.1f}x")
    
    # Outcome comparison
    hyper_mortality = 100 * (hyper_patients["IS_DEAD"] == 1).sum() / len(hyper_patients)
    normal_mortality = 100 * (normal_patients["IS_DEAD"] == 1).sum() / len(normal_patients)
    
    log(f"\n  Outcome:")
    log(f"    Hypermutator mortality: {hyper_mortality:.1f}%")
    log(f"    Normal mortality: {normal_mortality:.1f}%")
    
    # Fisher exact test
    a = int(((df["is_hypermutator"] == 1) & (df["IS_DEAD"] == 1)).sum())
    b = int(((df["is_hypermutator"] == 1) & (df["IS_DEAD"] == 0)).sum())
    c = int(((df["is_hypermutator"] == 0) & (df["IS_DEAD"] == 1)).sum())
    d = int(((df["is_hypermutator"] == 0) & (df["IS_DEAD"] == 0)).sum())
    
    odds_ratio, p_val = stats.fisher_exact([[a, b], [c, d]])
    log(f"    Odds ratio: {odds_ratio:.3f}, p={p_val:.4f}")
    
    # Clinical characteristics
    log(f"\n  Clinical characteristics:")
    for col in ["age", "stage"]:
        if col in df.columns:
            if col == "age":
                hyper_val = hyper_patients[col].dropna().mean()
                normal_val = normal_patients[col].dropna().mean()
                log(f"    {col}: hyper={hyper_val:.1f}, normal={normal_val:.1f}")
            else:
                hyper_dist = hyper_patients[col].value_counts(normalize=True) * 100
                log(f"    {col} (hypermutators):")
                for stage, pct in hyper_dist.head(3).items():
                    log(f"      {stage}: {pct:.1f}%")
    
    hyper_stats = {
        'n_hypermutators': int(len(hyper_patients)),
        'hyper_mortality_pct': float(hyper_mortality),
        'normal_mortality_pct': float(normal_mortality),
        'odds_ratio': float(odds_ratio),
        'p_value': float(p_val)
    }
    
    return hyper_stats

# ============================================================================
# PHASE 5: CONFOUNDING ANALYSIS
# ============================================================================

def analyze_confounding(df):
    """Analyze potential confounding structure."""
    section_header("PHASE 5: CONFOUNDING STRUCTURE")
    
    log("🔗 Checking confounder relationships...")
    
    confound_stats = {}
    
    # === Age vs Stage ===
    log("\n  Age vs Stage:")
    for stage in df["stage"].unique():
        stage_ages = df.loc[df["stage"] == stage, "age"].dropna()
        if len(stage_ages) > 0:
            log(f"    {stage}: mean age = {stage_ages.mean():.1f}")
    
    # ANOVA
    stage_groups = [df.loc[df["stage"] == stage, "age"].dropna() for stage in df["stage"].unique()]
    f_stat, p_val = stats.f_oneway(*stage_groups)
    log(f"  ANOVA: F={f_stat:.3f}, p={p_val:.4f}")
    
    confound_stats['age_vs_stage'] = {
        'f_statistic': float(f_stat),
        'p_value': float(p_val)
    }
    
    # === Stage vs Diagnosis ===
    log("\n  Stage vs Diagnosis (top 3):")
    top_diag = df["diagnosis"].value_counts().head(3).index
    for diag in top_diag:
        diag_stages = df.loc[df["diagnosis"] == diag, "stage"].value_counts(normalize=True) * 100
        log(f"    {diag[:40]}:")
        for stage, pct in diag_stages.head(3).items():
            log(f"      {stage}: {pct:.1f}%")
    
    return confound_stats

# ============================================================================
# PHASE 6: GENERATE VISUALIZATIONS
# ============================================================================

def generate_eda_plots(df, gene_cols, top_genes, gene_outcome_stats):
    """Generate comprehensive EDA plots."""
    section_header("PHASE 6: GENERATE EDA VISUALIZATIONS")
    
    log("Creating EDA plots...")
    
    # === Plot 1: Clinical Features vs Outcome ===
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 10))
    
    # Age vs Outcome
    age_alive = df.loc[df["IS_DEAD"] == 0, "age"].dropna()
    age_dead = df.loc[df["IS_DEAD"] == 1, "age"].dropna()
    
    axes1[0, 0].hist([age_alive, age_dead], bins=20, label=['Alive', 'Dead'], 
                     color=['green', 'red'], alpha=0.6, edgecolor='black')
    axes1[0, 0].set_xlabel('Age (years)', fontsize=11)
    axes1[0, 0].set_ylabel('Count', fontsize=11)
    axes1[0, 0].set_title('Age Distribution by Outcome', fontsize=12, fontweight='bold')
    axes1[0, 0].legend()
    axes1[0, 0].grid(alpha=0.3)
    
    # Stage vs Outcome
    stage_outcome = pd.crosstab(df["stage"], df["IS_DEAD"], normalize='index') * 100
    stage_outcome.plot(kind='bar', stacked=False, ax=axes1[0, 1], 
                       color=['green', 'red'], alpha=0.7, edgecolor='black')
    axes1[0, 1].set_xlabel('Stage', fontsize=11)
    axes1[0, 1].set_ylabel('Percentage (%)', fontsize=11)
    axes1[0, 1].set_title('Outcome by Stage', fontsize=12, fontweight='bold')
    axes1[0, 1].legend(['Alive', 'Dead'])
    axes1[0, 1].tick_params(axis='x', rotation=45)
    axes1[0, 1].grid(alpha=0.3)
    
    # Mutation burden vs Outcome
    normal_burden = df.loc[df["is_hypermutator"] == 0, gene_cols].sum(axis=1)
    hyper_burden = df.loc[df["is_hypermutator"] == 1, gene_cols].sum(axis=1)
    
    axes1[1, 0].hist([normal_burden[df.loc[df["is_hypermutator"]==0, "IS_DEAD"]==0],
                      normal_burden[df.loc[df["is_hypermutator"]==0, "IS_DEAD"]==1]],
                     bins=30, label=['Alive (normal)', 'Dead (normal)'], 
                     color=['green', 'red'], alpha=0.6, edgecolor='black')
    if len(hyper_burden) > 0:
        axes1[1, 0].axvline(hyper_burden.min(), color='purple', linestyle='--', linewidth=2, label='Hypermutator range')
    axes1[1, 0].set_xlabel('Mutation Burden', fontsize=11)
    axes1[1, 0].set_ylabel('Count', fontsize=11)
    axes1[1, 0].set_title('Mutation Burden by Outcome', fontsize=12, fontweight='bold')
    axes1[1, 0].legend(fontsize=9)
    axes1[1, 0].grid(alpha=0.3)
    
    # Top genes odds ratio
    gene_stats_df = pd.DataFrame(gene_outcome_stats).sort_values('odds_ratio', ascending=False)
    y_pos = np.arange(len(gene_stats_df))
    colors = ['red' if p < 0.05 else 'gray' for p in gene_stats_df['p_value']]
    
    axes1[1, 1].barh(y_pos, gene_stats_df['odds_ratio'], color=colors, edgecolor='black', alpha=0.7)
    axes1[1, 1].axvline(1, color='black', linestyle='--', linewidth=1)
    axes1[1, 1].set_yticks(y_pos)
    axes1[1, 1].set_yticklabels(gene_stats_df['gene'], fontsize=9)
    axes1[1, 1].set_xlabel('Odds Ratio', fontsize=11)
    axes1[1, 1].set_title('Top Genes: Odds Ratio for Mortality\n(red=p<0.05)', fontsize=12, fontweight='bold')
    axes1[1, 1].invert_yaxis()
    axes1[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    path1 = os.path.join(FIG_DIR, "clinical_outcome_analysis.png")
    plt.savefig(path1, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  ✓ Saved: {path1}")
    
    # === Plot 2: Top Genes Heatmap ===
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    
    # Select top 20 genes and sample 100 patients
    np.random.seed(42)
    sample_patients = np.random.choice(df.index, min(100, len(df)), replace=False)
    heatmap_data = df.loc[sample_patients, top_genes.index[:20]].T
    
    sns.heatmap(heatmap_data, cmap='RdYlGn_r', cbar_kws={'label': 'Mutation Status'}, 
                ax=ax2, linewidths=0.5, linecolor='gray')
    ax2.set_xlabel('Patient Index (sample)', fontsize=11)
    ax2.set_ylabel('Gene', fontsize=11)
    ax2.set_title('Top 20 Genes Mutation Pattern (100 patients sample)', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    path2 = os.path.join(FIG_DIR, "top_genes_heatmap.png")
    plt.savefig(path2, dpi=150, bbox_inches='tight')
    plt.close()
    log(f"  ✓ Saved: {path2}")
    
    log(f"\n✓ All EDA visualizations saved to {FIG_DIR}/")

# ============================================================================
# PHASE 7: SAVE OUTPUTS
# ============================================================================

def save_outputs(all_stats):
    """Save EDA outputs."""
    section_header("PHASE 7: SAVE OUTPUTS")
    
    # Save statistics JSON
    with open(EDA_STATS, 'w') as f:
        json.dump(all_stats, f, indent=2)
    log(f"✓ Statistics saved: {EDA_STATS}")
    
    # Save text report
    with open(EDA_REPORT, 'w') as f:
        f.write('\n'.join(LOG))
    log(f"✓ Report saved: {EDA_REPORT}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution."""
    section_header("EXPLORATORY DATA ANALYSIS PIPELINE")
    log("🔍 Starting EDA - Foundation for causal discovery...")
    
    # Phase 1: Load data
    df, gene_cols = load_data()
    
    # Phase 2: Clinical vs outcome
    clinical_stats = analyze_clinical_vs_outcome(df)
    
    # Phase 3: Top genes
    gene_outcome_stats, top_genes = analyze_top_genes(df, gene_cols, top_n=20)
    
    # Phase 4: Hypermutators
    hyper_stats = analyze_hypermutators(df, gene_cols)
    
    # Phase 5: Confounding
    confound_stats = analyze_confounding(df)
    
    # Phase 6: Visualizations
    generate_eda_plots(df, gene_cols, top_genes, gene_outcome_stats)
    
    # Compile all stats
    all_stats = {
        'timestamp': datetime.now().isoformat(),
        'dataset': {
            'n_patients': int(len(df)),
            'n_genes': int(len(gene_cols)),
            'n_dead': int((df['IS_DEAD']==1).sum()),
            'n_alive': int((df['IS_DEAD']==0).sum())
        },
        'clinical_vs_outcome': clinical_stats,
        'hypermutator_analysis': hyper_stats,
        'confounding_analysis': confound_stats,
        'top_gene_associations': gene_outcome_stats
    }
    
    # Phase 7: Save outputs
    save_outputs(all_stats)
    
    # Final summary
    section_header("✅ EDA COMPLETE")
    log("Key findings summary:")
    log(f"  📁 Report: {EDA_REPORT}")
    log(f"  📁 Statistics: {EDA_STATS}")
    log(f"  📁 Figures: {FIG_DIR}/")
    
    log(f"\n🎯 NEXT STEPS:")
    log(f"  1. Review EDA visualizations in {FIG_DIR}/")
    log(f"  2. Check {EDA_STATS} for quantitative results")
    log(f"  3. Proceed to causal discovery: python 03_causal_discovery.py")
    
    log("\n🎉 EDA complete! Ready for causal modeling.")


if __name__ == "__main__":
    main()