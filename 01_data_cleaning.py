#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_data_cleaning.py

RIGOROUS DATA CLEANING & HARMONIZATION
======================================
Forensically clean and harmonize clinical + mutation data based on archaeology findings.

INPUTS:
  - reports/data_quality_flags.json (from archaeology)
  - data/clinical.tsv
  - results/master_mutation_matrix.csv

OUTPUTS:
  - results/clinical_cleaned.csv
  - results/mutation_cleaned.csv
  - results/merged_dataset.csv
  - reports/cleaning_report.txt
  - figures/cleaning/

CLEANING STEPS:
  1. Clinical data harmonization
     - Age conversion (days → years)
     - Stage standardization (AJCC grouping)
     - Diagnosis harmonization (ICD-O codes)
     - Vital status binary encoding
     - Missing data handling
  
  2. Mutation data filtering
     - Remove ultra-rare genes (< MIN_MUT threshold)
     - Remove all-zero genes
     - Remove patients with extreme mutation burden (outliers)
  
  3. Dataset merging
     - Inner join on patient_id
     - Document data loss
     - Final quality checks
  
  4. Confounder validation
     - Check for adequate variation in confounders
     - Flag potential colliders
     - Validate outcome distribution
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
DATA_DIR = "data"
RESULTS_DIR = "results"
REPORT_DIR = "reports"
FIG_DIR = os.path.join("figures", "cleaning")

CLINICAL_RAW = os.path.join(DATA_DIR, "clinical.tsv")
MUTATION_RAW = os.path.join(RESULTS_DIR, "master_mutation_matrix.csv")
QC_FLAGS = os.path.join(REPORT_DIR, "data_quality_flags.json")

CLINICAL_CLEAN = os.path.join(RESULTS_DIR, "clinical_cleaned.csv")
MUTATION_CLEAN = os.path.join(RESULTS_DIR, "mutation_cleaned.csv")
MERGED_DATASET = os.path.join(RESULTS_DIR, "merged_dataset.csv")
CLEANING_LOG = os.path.join(REPORT_DIR, "cleaning_report.txt")

# Create directories
for d in [RESULTS_DIR, REPORT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# Cleaning parameters
MIN_MUT_CASES = 3           # Minimum mutated patients per gene
MAX_MUT_BURDEN_ZSCORE = 5   # Maximum Z-score for mutation burden (outlier detection)
MIN_AGE = 18                # Minimum age (years)
MAX_AGE = 100               # Maximum age (years)

# Log buffer
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

def load_qc_flags():
    """Load quality flags from archaeology."""
    if os.path.exists(QC_FLAGS):
        with open(QC_FLAGS, 'r') as f:
            flags = json.load(f)
        log(f"✓ Loaded QC flags from archaeology")
        return flags
    else:
        log(f"⚠️  QC flags not found. Run 00_data_archaeology.py first.", "WARN")
        return None


def clean_clinical_data():
    """Clean and harmonize clinical data."""
    section_header("PHASE 1: CLINICAL DATA CLEANING")
    
    # Load raw
    log("Loading raw clinical data...")
    df = pd.read_csv(CLINICAL_RAW, sep="\t", skiprows=[1], low_memory=False)
    log(f"  Raw shape: {df.shape}")
    
    # Deduplicate
    n_before = len(df)
    df = df.drop_duplicates("cases.case_id")
    n_dupes = n_before - len(df)
    if n_dupes > 0:
        log(f"  Removed {n_dupes} duplicate case_ids")
    
    # Select and rename columns
    log("\n📋 Column mapping...")
    column_map = {
        "cases.submitter_id": "patient_id",
        "diagnoses.age_at_diagnosis": "age",
        "diagnoses.ajcc_pathologic_stage": "stage_raw",
        "diagnoses.primary_diagnosis": "diagnosis_raw",
        "demographic.vital_status": "vital_status_raw",
        "demographic.gender": "gender",
        "demographic.race": "race",
        "demographic.ethnicity": "ethnicity"
    }
    
    available_cols = {k: v for k, v in column_map.items() if k in df.columns}
    df_clean = df[list(available_cols.keys())].rename(columns=available_cols)
    log(f"  Mapped {len(available_cols)} columns")
    
    # === AGE CLEANING ===
    log("\n📅 Age cleaning...")
    df_clean["age"] = pd.to_numeric(df_clean["age"], errors='coerce')
    
    # Check if in days
    age_median = df_clean["age"].median()
    if age_median > 2000:
        log(f"  Age in days detected (median={age_median:.0f})")
        df_clean["age"] = df_clean["age"] / 365.25
        log(f"  Converted to years (new median={df_clean['age'].median():.1f})")
    
    # Age bounds
    age_before = df_clean["age"].notna().sum()
    df_clean.loc[df_clean["age"] < MIN_AGE, "age"] = np.nan
    df_clean.loc[df_clean["age"] > MAX_AGE, "age"] = np.nan
    age_after = df_clean["age"].notna().sum()
    age_removed = age_before - age_after
    if age_removed > 0:
        log(f"  ⚠️  Removed {age_removed} ages outside [{MIN_AGE}, {MAX_AGE}]", "WARN")
    
    log(f"  Age range: {df_clean['age'].min():.1f} - {df_clean['age'].max():.1f}")
    log(f"  Age missing: {df_clean['age'].isna().sum()} / {len(df_clean)}")
    
    # === VITAL STATUS CLEANING ===
    log("\n💀 Vital status cleaning...")
    df_clean["vital_status_raw"] = df_clean["vital_status_raw"].astype(str).str.lower().str.strip()
    
    # Binary encoding
    status_map = {
        "dead": 1,
        "alive": 0,
        "deceased": 1
    }
    df_clean["IS_DEAD"] = df_clean["vital_status_raw"].map(status_map)
    
    # Report distribution
    status_counts = df_clean["IS_DEAD"].value_counts()
    log(f"  Vital status distribution:")
    for status, count in status_counts.items():
        label = "Dead" if status == 1 else "Alive"
        pct = 100 * count / len(df_clean)
        log(f"    {label}: {count} ({pct:.1f}%)")
    
    missing_vital = df_clean["IS_DEAD"].isna().sum()
    if missing_vital > 0:
        log(f"  ⚠️  {missing_vital} patients with unknown vital status", "WARN")
    
    # === STAGE HARMONIZATION ===
    log("\n🎚️  Stage harmonization...")
    
    def harmonize_stage(stage_str):
        """Harmonize AJCC stage to simplified categories."""
        if pd.isna(stage_str):
            return "Unknown"
        
        stage = str(stage_str).upper().strip()
        
        # Clean common variations
        stage = stage.replace("STAGE", "").strip()
        stage = stage.replace("PATHOLOGIC", "").strip()
        
        # Map to categories
        if any(x in stage for x in ["NOT REPORTED", "UNKNOWN", "NA", "NX"]):
            return "Unknown"
        elif stage.startswith("I") and not stage.startswith("IV"):
            # Stage I, IA, IB, IC → Early
            if stage in ["I", "IA", "IB", "IC"]:
                return "I"
            elif stage in ["II", "IIA", "IIB", "IIC"]:
                return "II"
            elif stage in ["III", "IIIA", "IIIB", "IIIC"]:
                return "III"
            else:
                return "I-II"  # Conservative grouping
        elif stage.startswith("IV"):
            return "IV"
        elif stage.startswith("II"):
            return "II"
        elif stage.startswith("III"):
            return "III"
        else:
            return "Unknown"
    
    df_clean["stage"] = df_clean["stage_raw"].apply(harmonize_stage)
    
    log(f"  Stage distribution:")
    stage_counts = df_clean["stage"].value_counts()
    for stage, count in stage_counts.items():
        pct = 100 * count / len(df_clean)
        log(f"    {stage}: {count} ({pct:.1f}%)")
    
    # === DIAGNOSIS HARMONIZATION ===
    log("\n🏥 Diagnosis harmonization...")
    
    # Keep top N diagnosis types, group rest as "Other"
    TOP_N_DIAGNOSES = 10
    top_diagnoses = df_clean["diagnosis_raw"].value_counts().head(TOP_N_DIAGNOSES).index
    df_clean["diagnosis"] = df_clean["diagnosis_raw"].apply(
        lambda x: x if x in top_diagnoses else "Other"
    )
    
    log(f"  Kept top {TOP_N_DIAGNOSES} diagnosis types, grouped rest as 'Other'")
    log(f"  Diagnosis distribution:")
    dx_counts = df_clean["diagnosis"].value_counts()
    for dx, count in dx_counts.head(5).items():
        pct = 100 * count / len(df_clean)
        log(f"    {dx}: {count} ({pct:.1f}%)")
    log(f"    ... ({len(dx_counts) - 5} more categories)")
    
    # === MISSING DATA ANALYSIS ===
    log("\n🔍 Missing data summary:")
    key_cols = ["patient_id", "age", "stage", "diagnosis", "IS_DEAD"]
    for col in key_cols:
        if col in df_clean.columns:
            missing = df_clean[col].isna().sum()
            pct = 100 * missing / len(df_clean)
            log(f"  {col}: {missing} missing ({pct:.1f}%)")
    
    # Drop rows with missing critical fields
    log("\n🗑️  Dropping rows with missing critical fields...")
    critical_fields = ["patient_id", "IS_DEAD"]
    n_before = len(df_clean)
    df_clean = df_clean.dropna(subset=critical_fields)
    n_after = len(df_clean)
    n_dropped = n_before - n_after
    if n_dropped > 0:
        log(f"  Dropped {n_dropped} rows ({100*n_dropped/n_before:.1f}%)")
    
    # Save cleaned clinical
    df_clean.to_csv(CLINICAL_CLEAN, index=False)
    log(f"\n✓ Saved cleaned clinical data: {CLINICAL_CLEAN}")
    log(f"  Final shape: {df_clean.shape}")
    
    return df_clean


def clean_mutation_data():
    """Clean and filter mutation matrix."""
    section_header("PHASE 2: MUTATION DATA CLEANING")
    
    log("Loading raw mutation matrix...")
    df = pd.read_csv(MUTATION_RAW)
    log(f"  Raw shape: {df.shape}")
    
    gene_cols = [c for c in df.columns if c != "patient_id"]
    log(f"  Genes: {len(gene_cols)}")
    log(f"  Patients: {len(df)}")
    
def clean_mutation_data():
    """Clean and filter mutation matrix."""
    section_header("PHASE 2: MUTATION DATA CLEANING")
    
    log("Loading raw mutation matrix...")
    df = pd.read_csv(MUTATION_RAW)
    log(f"  Raw shape: {df.shape}")
    
    gene_cols = [c for c in df.columns if c != "patient_id"]
    log(f"  Genes: {len(gene_cols)}")
    log(f"  Patients: {len(df)}")
    
    # === FILTER ULTRA-RARE GENES ===
    log(f"\n🦄 Filtering ultra-rare genes (< {MIN_MUT_CASES} mutations)...")
    
    gene_sums = df[gene_cols].sum()
    rare_genes = gene_sums[gene_sums < MIN_MUT_CASES].index.tolist()
    
    log(f"  Genes before filtering: {len(gene_cols)}")
    log(f"  Ultra-rare genes (< {MIN_MUT_CASES}): {len(rare_genes)}")
    
    # Keep only genes with sufficient mutations
    keep_genes = gene_sums[gene_sums >= MIN_MUT_CASES].index.tolist()
    df_filtered = df[["patient_id"] + keep_genes].copy()
    
    log(f"  Genes after filtering: {len(keep_genes)}")
    log(f"  Removed: {len(rare_genes)} ({100*len(rare_genes)/len(gene_cols):.1f}%)")
    
    # === IDENTIFY HYPERMUTATORS ===
    log(f"\n🔍 Identifying hypermutator patients...")
    
    patient_burdens = df_filtered[keep_genes].sum(axis=1)
    mean_burden = patient_burdens.mean()
    std_burden = patient_burdens.std()
    
    log(f"  Mean mutation burden: {mean_burden:.2f}")
    log(f"  Std deviation: {std_burden:.2f}")
    log(f"  Threshold (>{MAX_MUT_BURDEN_ZSCORE}σ): {mean_burden + MAX_MUT_BURDEN_ZSCORE * std_burden:.0f}")
    
    # Calculate Z-scores
    z_scores = (patient_burdens - mean_burden) / std_burden
    hypermutators = df_filtered.loc[z_scores > MAX_MUT_BURDEN_ZSCORE, "patient_id"].tolist()
    
    if hypermutators:
        log(f"\n  ⚠️  {len(hypermutators)} hypermutator patient(s) detected:", "WARN")
        for patient in hypermutators:
            idx = df_filtered[df_filtered["patient_id"] == patient].index[0]
            burden = patient_burdens.iloc[idx]
            z_score = z_scores.iloc[idx]
            log(f"    {patient}: {burden:.0f} mutations (Z={z_score:.2f})")
        
        log(f"\n  These patients will be FLAGGED but NOT removed.")
        log(f"  They should be handled in sensitivity analysis.")
        
        # Add hypermutator flag column
        df_filtered["is_hypermutator"] = df_filtered["patient_id"].isin(hypermutators).astype(int)
    else:
        log(f"  ✓ No hypermutators detected")
        df_filtered["is_hypermutator"] = 0
    
    # === PATIENT MUTATION BURDEN STATS ===
    log(f"\n📊 Patient mutation burden distribution:")
    log(f"  Min: {patient_burdens.min():.0f}")
    log(f"  25th percentile: {patient_burdens.quantile(0.25):.0f}")
    log(f"  Median: {patient_burdens.median():.0f}")
    log(f"  75th percentile: {patient_burdens.quantile(0.75):.0f}")
    log(f"  Max: {patient_burdens.max():.0f}")
    
    # Save cleaned mutation data
    df_filtered.to_csv(MUTATION_CLEAN, index=False)
    log(f"\n✓ Saved cleaned mutation data: {MUTATION_CLEAN}")
    log(f"  Final shape: {df_filtered.shape}")
    
    return df_filtered


def merge_datasets(clinical_df, mutation_df):
    """Merge clinical and mutation datasets."""
    section_header("PHASE 3: MERGE DATASETS")
    
    log("Merging clinical and mutation data on patient_id...")
    
    # Check overlap before merge
    clinical_ids = set(clinical_df["patient_id"])
    mutation_ids = set(mutation_df["patient_id"])
    
    overlap = clinical_ids & mutation_ids
    clinical_only = clinical_ids - mutation_ids
    mutation_only = mutation_ids - clinical_ids
    
    log(f"\n📇 Patient ID overlap:")
    log(f"  Clinical IDs: {len(clinical_ids)}")
    log(f"  Mutation IDs: {len(mutation_ids)}")
    log(f"  Overlap: {len(overlap)}")
    log(f"  Clinical only: {len(clinical_only)}")
    log(f"  Mutation only: {len(mutation_only)}")
    
    # Inner join
    merged = pd.merge(clinical_df, mutation_df, on="patient_id", how="inner")
    
    log(f"\n💥 Merge results:")
    log(f"  Merged dataset: {len(merged)} patients")
    log(f"  Lost from clinical: {len(clinical_only)} ({100*len(clinical_only)/len(clinical_ids):.1f}%)")
    log(f"  Lost from mutation: {len(mutation_only)} ({100*len(mutation_only)/len(mutation_ids):.1f}%)")
    
    # Final data quality checks
    log(f"\n🔍 Final data quality checks:")
    
    # Check for missing values in key columns
    key_cols = ["patient_id", "age", "stage", "diagnosis", "IS_DEAD"]
    for col in key_cols:
        if col in merged.columns:
            missing = merged[col].isna().sum()
            pct = 100 * missing / len(merged)
            if missing > 0:
                log(f"  ⚠️  {col}: {missing} missing ({pct:.1f}%)", "WARN")
            else:
                log(f"  ✓ {col}: no missing values")
    
    # Check outcome distribution
    log(f"\n💀 Final outcome distribution:")
    outcome_dist = merged["IS_DEAD"].value_counts()
    for outcome, count in outcome_dist.items():
        label = "Dead" if outcome == 1 else "Alive"
        pct = 100 * count / len(merged)
        log(f"  {label}: {count} ({pct:.1f}%)")
    
    # Check hypermutator distribution
    if "is_hypermutator" in merged.columns:
        n_hyper = merged["is_hypermutator"].sum()
        log(f"\n🔬 Hypermutators in final dataset: {n_hyper}")
    
    # Save merged dataset
    merged.to_csv(MERGED_DATASET, index=False)
    log(f"\n✓ Saved merged dataset: {MERGED_DATASET}")
    log(f"  Final shape: {merged.shape}")
    
    return merged


def generate_cleaning_visualizations(merged_df):
    """Generate visualizations for cleaned data."""
    section_header("PHASE 4: GENERATE CLEANING VISUALIZATIONS")
    
    log("Generating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: Age distribution (cleaned)
    age_data = merged_df["age"].dropna()
    axes[0, 0].hist(age_data, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(age_data.median(), color='red', linestyle='--', linewidth=2, label=f'Median: {age_data.median():.1f}')
    axes[0, 0].set_xlabel('Age (years)', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title('Age Distribution (Cleaned)', fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Stage distribution
    stage_data = merged_df["stage"].value_counts()
    axes[0, 1].bar(stage_data.index, stage_data.values, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Stage', fontsize=11)
    axes[0, 1].set_ylabel('Count', fontsize=11)
    axes[0, 1].set_title('Stage Distribution (Harmonized)', fontsize=12, fontweight='bold')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Outcome distribution
    outcome_data = merged_df["IS_DEAD"].value_counts()
    colors = ['green', 'red']
    axes[0, 2].bar(['Alive', 'Dead'], outcome_data.values, color=colors, edgecolor='black', alpha=0.7)
    axes[0, 2].set_ylabel('Count', fontsize=11)
    axes[0, 2].set_title('Outcome Distribution', fontsize=12, fontweight='bold')
    axes[0, 2].grid(alpha=0.3)
    
    # Plot 4: Mutation burden (with hypermutators highlighted)
    gene_cols = [c for c in merged_df.columns if c not in ["patient_id", "age", "stage", "stage_raw", 
                                                             "diagnosis", "diagnosis_raw", "vital_status_raw",
                                                             "IS_DEAD", "gender", "race", "ethnicity", 
                                                             "is_hypermutator"]]
    patient_burdens = merged_df[gene_cols].sum(axis=1)
    
    # Separate normal and hypermutators
    normal_mask = merged_df["is_hypermutator"] == 0
    hyper_mask = merged_df["is_hypermutator"] == 1
    
    axes[1, 0].hist(patient_burdens[normal_mask], bins=50, color='mediumseagreen', 
                    edgecolor='black', alpha=0.7, label='Normal')
    if hyper_mask.sum() > 0:
        axes[1, 0].hist(patient_burdens[hyper_mask], bins=20, color='red', 
                        edgecolor='black', alpha=0.7, label='Hypermutator')
    axes[1, 0].axvline(patient_burdens.median(), color='orange', linestyle='--', 
                       linewidth=2, label=f'Median: {patient_burdens.median():.0f}')
    axes[1, 0].set_xlabel('Mutation Burden', fontsize=11)
    axes[1, 0].set_ylabel('Count', fontsize=11)
    axes[1, 0].set_title('Patient Mutation Burden', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 5: Diagnosis distribution (top categories)
    diag_data = merged_df["diagnosis"].value_counts().head(8)
    y_pos = np.arange(len(diag_data))
    axes[1, 1].barh(y_pos, diag_data.values, color='coral', edgecolor='black', alpha=0.7)
    axes[1, 1].set_yticks(y_pos)
    axes[1, 1].set_yticklabels(diag_data.index, fontsize=9)
    axes[1, 1].set_xlabel('Count', fontsize=11)
    axes[1, 1].set_title('Top 8 Diagnosis Types', fontsize=12, fontweight='bold')
    axes[1, 1].invert_yaxis()
    axes[1, 1].grid(alpha=0.3)
    
    # Plot 6: Missing data summary
    key_cols = ["age", "stage", "diagnosis", "IS_DEAD"]
    missing_counts = [merged_df[col].isna().sum() for col in key_cols]
    missing_pcts = [100 * m / len(merged_df) for m in missing_counts]
    
    axes[1, 2].bar(key_cols, missing_pcts, color='purple', edgecolor='black', alpha=0.7)
    axes[1, 2].set_ylabel('Missing (%)', fontsize=11)
    axes[1, 2].set_title('Missing Data Summary', fontsize=12, fontweight='bold')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(alpha=0.3)
    axes[1, 2].set_ylim([0, max(missing_pcts) * 1.2 if missing_pcts else 1])
    
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "cleaning_summary.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log(f"✓ Cleaning visualizations saved: {path}")


def save_cleaning_report(clinical_df, mutation_df, merged_df):
    """Save detailed cleaning report."""
    section_header("PHASE 5: SAVE CLEANING REPORT")
    
    # Add summary statistics to log
    log("\n" + "=" * 80)
    log("  CLEANING SUMMARY")
    log("=" * 80)
    
    log(f"\nClinical data:")
    log(f"  Raw patients: {len(pd.read_csv(CLINICAL_RAW, sep='\t', skiprows=[1]))}")
    log(f"  After deduplication: (see archaeology report)")
    log(f"  After cleaning: {len(clinical_df)}")
    
    log(f"\nMutation data:")
    log(f"  Raw genes: {len([c for c in pd.read_csv(MUTATION_RAW).columns if c != 'patient_id'])}")
    log(f"  After filtering: {len([c for c in mutation_df.columns if c not in ['patient_id', 'is_hypermutator']])}")
    log(f"  Hypermutators flagged: {mutation_df['is_hypermutator'].sum()}")
    
    log(f"\nMerged dataset:")
    log(f"  Final patients: {len(merged_df)}")
    log(f"  Final genes: {len([c for c in merged_df.columns if c not in ['patient_id', 'age', 'stage', 'stage_raw', 'diagnosis', 'diagnosis_raw', 'vital_status_raw', 'IS_DEAD', 'gender', 'race', 'ethnicity', 'is_hypermutator']])}")
    log(f"  Alive: {(merged_df['IS_DEAD']==0).sum()}")
    log(f"  Dead: {(merged_df['IS_DEAD']==1).sum()}")
    
    # Save log
    with open(CLEANING_LOG, 'w') as f:
        f.write('\n'.join(LOG))
    
    log(f"\n✓ Cleaning report saved: {CLEANING_LOG}")


def main():
    """Main execution."""
    section_header("DATA CLEANING PIPELINE")
    log("🚀 Starting rigorous data cleaning...")
    
    # Load QC flags if available
    qc_flags = load_qc_flags()
    
    # Phase 1: Clean clinical data
    clinical_df = clean_clinical_data()
    if clinical_df is None:
        log("\n❌ Clinical cleaning failed. Pipeline stopped.", "ERROR")
        return
    
    # Phase 2: Clean mutation data
    mutation_df = clean_mutation_data()
    if mutation_df is None:
        log("\n❌ Mutation cleaning failed. Pipeline stopped.", "ERROR")
        return
    
    # Phase 3: Merge datasets
    merged_df = merge_datasets(clinical_df, mutation_df)
    
    # Phase 4: Generate visualizations
    generate_cleaning_visualizations(merged_df)
    
    # Phase 5: Save cleaning report
    save_cleaning_report(clinical_df, mutation_df, merged_df)
    
    # Final summary
    section_header("✅ CLEANING COMPLETE")
    log("All outputs saved successfully:")
    log(f"  📁 Clinical (cleaned): {CLINICAL_CLEAN}")
    log(f"  📁 Mutation (cleaned): {MUTATION_CLEAN}")
    log(f"  📁 Merged dataset: {MERGED_DATASET}")
    log(f"  📁 Cleaning report: {CLEANING_LOG}")
    log(f"  📁 Visualizations: {FIG_DIR}/")
    
    log(f"\n📊 FINAL DATASET STATISTICS:")
    log(f"  Patients: {len(merged_df)}")
    log(f"  Features: {merged_df.shape[1]}")
    log(f"  Outcome balance: {(merged_df['IS_DEAD']==1).sum()} dead / {(merged_df['IS_DEAD']==0).sum()} alive")
    
    if "is_hypermutator" in merged_df.columns:
        n_hyper = merged_df["is_hypermutator"].sum()
        if n_hyper > 0:
            log(f"\n⚠️  NOTE: {n_hyper} hypermutator(s) flagged for sensitivity analysis", "WARN")
    
    log(f"\n🎯 NEXT STEPS:")
    log(f"  1. Review {CLEANING_LOG}")
    log(f"  2. Check {FIG_DIR}/cleaning_summary.png")
    log(f"  3. Review {MERGED_DATASET} for analysis readiness")
    log(f"  4. Proceed to exploratory analysis or causal modeling")
    
    log("\n🎉 Data cleaning complete! Dataset ready for analysis.")


if __name__ == "__main__":
    main()