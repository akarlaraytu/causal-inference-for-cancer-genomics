#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00_data_archaeology.py

FORENSIC DATA INSPECTION
========================
Veri setlerinin anatomisi, kalitesi, ve yapısını anlamak için detaylı inceleme.

INPUTS:
  - data/clinical.tsv (TCGA raw clinical data)
  - results/master_mutation_matrix.csv (generated mutation matrix)

OUTPUTS:
  - reports/data_archaeology_report.txt
  - reports/data_quality_flags.json
  - figures/archaeology/
    - clinical_structure.png
    - mutation_sparsity.png
    - patient_overlap.png
    - missing_pattern.png

Bu script hiçbir veri değiştirmez, sadece inceler ve raporlar.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime

warnings.filterwarnings('ignore')

# Paths
DATA_DIR = "data"
RESULTS_DIR = "results"
REPORT_DIR = "reports"
FIG_DIR = os.path.join("figures", "archaeology")

CLINICAL_FILE = os.path.join(DATA_DIR, "clinical.tsv")
MUTATION_FILE = os.path.join(RESULTS_DIR, "master_mutation_matrix.csv")

# Create output directories
for d in [REPORT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# Global report storage
REPORT = []

def log(msg, level="INFO"):
    """Log message to console and report buffer."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] [{level}] {msg}"
    print(formatted)
    REPORT.append(formatted)

def section_header(title):
    """Print section header."""
    border = "=" * 80
    log(f"\n{border}")
    log(f"  {title}")
    log(f"{border}\n")

def inspect_clinical_raw():
    """Deep inspection of clinical.tsv file."""
    section_header("PHASE 1: CLINICAL DATA ARCHAEOLOGY")
    
    # Read with minimal processing
    log("Reading clinical.tsv (raw)...")
    
    # First, peek at first few lines to understand structure
    with open(CLINICAL_FILE, 'r') as f:
        first_lines = [f.readline() for _ in range(5)]
    
    log("First 5 lines of clinical.tsv:")
    for i, line in enumerate(first_lines):
        log(f"  Line {i}: {line[:100]}...")
    
    # Try reading with different configurations
    try:
        # Attempt 1: With second row as header (common in TCGA)
        df_skip1 = pd.read_csv(CLINICAL_FILE, sep="\t", skiprows=[1], low_memory=False)
        log(f"✓ Read with skiprows=[1]: shape={df_skip1.shape}")
        
        # Attempt 2: No skip
        df_noskip = pd.read_csv(CLINICAL_FILE, sep="\t", low_memory=False)
        log(f"✓ Read without skip: shape={df_noskip.shape}")
        
        # Compare
        if df_skip1.shape != df_noskip.shape:
            log("⚠️  WARNING: Different shapes with/without skiprows!", "WARN")
            log(f"   Difference: {df_noskip.shape[0] - df_skip1.shape[0]} rows")
        
        # Use the version with skiprows (standard for TCGA)
        df = df_skip1
        
    except Exception as e:
        log(f"✗ Error reading clinical file: {e}", "ERROR")
        return None
    
    # Basic statistics
    log(f"\n📊 CLINICAL DATA OVERVIEW:")
    log(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    log(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    # Column analysis
    log(f"\n📋 COLUMN INVENTORY:")
    log(f"  Total columns: {len(df.columns)}")
    
    # Categorize columns by prefix
    prefixes = {}
    for col in df.columns:
        prefix = col.split('.')[0] if '.' in col else 'other'
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    
    log(f"\n  Column prefixes:")
    for prefix, count in sorted(prefixes.items(), key=lambda x: -x[1]):
        log(f"    {prefix}: {count} columns")
    
    # Key columns check
    key_columns = {
        "cases.case_id": "Patient identifier",
        "cases.submitter_id": "Patient submitter ID",
        "demographic.vital_status": "Survival status",
        "diagnoses.age_at_diagnosis": "Age at diagnosis",
        "diagnoses.ajcc_pathologic_stage": "Tumor stage",
        "diagnoses.primary_diagnosis": "Diagnosis type"
    }
    
    log(f"\n🔑 KEY COLUMNS CHECK:")
    missing_keys = []
    for col, desc in key_columns.items():
        if col in df.columns:
            non_null = df[col].notna().sum()
            pct = 100 * non_null / len(df)
            log(f"  ✓ {col}: {non_null}/{len(df)} ({pct:.1f}%) non-null")
        else:
            log(f"  ✗ {col}: NOT FOUND", "ERROR")
            missing_keys.append(col)
    
    if missing_keys:
        log(f"\n⚠️  Missing {len(missing_keys)} key columns!", "WARN")
        return None
    
    # Deduplicate by case_id
    log(f"\n🔍 DUPLICATE ANALYSIS:")
    n_before = len(df)
    df_dedup = df.drop_duplicates("cases.case_id")
    n_after = len(df_dedup)
    log(f"  Before dedup: {n_before} rows")
    log(f"  After dedup:  {n_after} rows")
    log(f"  Removed: {n_before - n_after} duplicates")
    
    # Vital status distribution
    log(f"\n💀 VITAL STATUS DISTRIBUTION:")
    vital = df_dedup["demographic.vital_status"].value_counts()
    for status, count in vital.items():
        pct = 100 * count / len(df_dedup)
        log(f"  {status}: {count} ({pct:.1f}%)")
    
    # Age distribution
    log(f"\n📅 AGE AT DIAGNOSIS:")
    age = pd.to_numeric(df_dedup["diagnoses.age_at_diagnosis"], errors='coerce')
    log(f"  Non-null: {age.notna().sum()} / {len(df_dedup)}")
    log(f"  Min: {age.min():.0f}")
    log(f"  Median: {age.median():.0f}")
    log(f"  Max: {age.max():.0f}")
    
    # Check if age is in days (TCGA sometimes stores in days)
    if age.median() > 2000:
        log(f"  ⚠️  Age appears to be in DAYS (median={age.median():.0f})", "WARN")
        log(f"  Converting to years: {age.median() / 365.25:.1f} years")
    
    # Stage distribution
    log(f"\n🎚️  STAGE DISTRIBUTION:")
    stage = df_dedup["diagnoses.ajcc_pathologic_stage"].value_counts().head(10)
    for st, count in stage.items():
        log(f"  {st}: {count}")
    
    # Diagnosis types
    log(f"\n🏥 DIAGNOSIS TYPES:")
    dx = df_dedup["diagnoses.primary_diagnosis"].value_counts().head(10)
    for d, count in dx.items():
        log(f"  {d}: {count}")
    
    # Save clinical summary
    df_dedup.to_csv(os.path.join(REPORT_DIR, "clinical_deduplicated.csv"), index=False)
    log(f"\n✓ Saved deduplicated clinical data to reports/")
    
    return df_dedup


def inspect_mutation_matrix():
    """Deep inspection of mutation matrix."""
    section_header("PHASE 2: MUTATION MATRIX ARCHAEOLOGY")
    
    if not os.path.exists(MUTATION_FILE):
        log(f"✗ Mutation file not found: {MUTATION_FILE}", "ERROR")
        log("  This file should be generated from MAF files.", "INFO")
        return None
    
    log("Reading mutation matrix...")
    df = pd.read_csv(MUTATION_FILE)
    
    log(f"\n📊 MUTATION MATRIX OVERVIEW:")
    log(f"  Shape: {df.shape[0]} patients × {df.shape[1]} genes")
    log(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB")
    
    # Check patient_id column
    if "patient_id" not in df.columns:
        log("✗ No patient_id column found!", "ERROR")
        return None
    
    log(f"  Patients: {len(df)}")
    log(f"  Genes: {df.shape[1] - 1}")  # -1 for patient_id
    
    # Gene mutation frequencies
    gene_cols = [c for c in df.columns if c != "patient_id"]
    mut_counts = df[gene_cols].sum().sort_values(ascending=False)
    
    log(f"\n🧬 MUTATION FREQUENCY STATS:")
    log(f"  Total mutations: {mut_counts.sum():.0f}")
    log(f"  Mean mutations per gene: {mut_counts.mean():.2f}")
    log(f"  Median mutations per gene: {mut_counts.median():.0f}")
    log(f"  Max mutations (single gene): {mut_counts.max():.0f}")
    
    # Sparsity analysis
    total_cells = len(df) * len(gene_cols)
    mut_cells = df[gene_cols].sum().sum()
    sparsity = 100 * (1 - mut_cells / total_cells)
    
    log(f"\n🕳️  SPARSITY ANALYSIS:")
    log(f"  Total cells: {total_cells:,}")
    log(f"  Mutated cells: {mut_cells:,}")
    log(f"  Sparsity: {sparsity:.2f}%")
    
    # Top mutated genes
    log(f"\n🔝 TOP 20 MUTATED GENES:")
    for i, (gene, count) in enumerate(mut_counts.head(20).items(), 1):
        pct = 100 * count / len(df)
        log(f"  {i:2d}. {gene}: {count:.0f} ({pct:.1f}%)")
    
    # Rare mutation analysis
    rare_thresholds = [1, 2, 5, 10, 20, 50]
    log(f"\n🦄 RARE MUTATION DISTRIBUTION:")
    for thresh in rare_thresholds:
        n_rare = (mut_counts <= thresh).sum()
        pct = 100 * n_rare / len(mut_counts)
        log(f"  Genes with ≤{thresh:2d} mutations: {n_rare:5d} ({pct:.1f}%)")
    
    # Per-patient mutation burden
    patient_burdens = df[gene_cols].sum(axis=1)
    log(f"\n👤 PATIENT MUTATION BURDEN:")
    log(f"  Min: {patient_burdens.min():.0f}")
    log(f"  25th percentile: {patient_burdens.quantile(0.25):.0f}")
    log(f"  Median: {patient_burdens.median():.0f}")
    log(f"  75th percentile: {patient_burdens.quantile(0.75):.0f}")
    log(f"  Max: {patient_burdens.max():.0f}")
    
    # Check for data quality issues
    log(f"\n🚨 DATA QUALITY CHECKS:")
    
    # 1. All zeros or all ones?
    all_zero_genes = [g for g in gene_cols if df[g].sum() == 0]
    all_one_genes = [g for g in gene_cols if df[g].sum() == len(df)]
    
    if all_zero_genes:
        log(f"  ⚠️  {len(all_zero_genes)} genes with ZERO mutations", "WARN")
    else:
        log(f"  ✓ No all-zero genes")
    
    if all_one_genes:
        log(f"  ⚠️  {len(all_one_genes)} genes mutated in ALL patients", "WARN")
    else:
        log(f"  ✓ No all-one genes")
    
    # 2. Non-binary values?
    non_binary = []
    for gene in gene_cols[:100]:  # Sample first 100
        unique_vals = df[gene].unique()
        if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            non_binary.append((gene, unique_vals))
    
    if non_binary:
        log(f"  ⚠️  Non-binary values detected in {len(non_binary)} genes!", "WARN")
        for gene, vals in non_binary[:5]:
            log(f"     {gene}: {vals}")
    else:
        log(f"  ✓ All checked genes are binary (0/1)")
    
    # 3. Missing patient IDs?
    null_ids = df["patient_id"].isna().sum()
    if null_ids > 0:
        log(f"  ⚠️  {null_ids} patients with NULL ID", "WARN")
    else:
        log(f"  ✓ No missing patient IDs")
    
    return df, mut_counts, patient_burdens


def cross_validate_datasets(clinical_df, mutation_df):
    """Check overlap and consistency between datasets."""
    section_header("PHASE 3: CROSS-DATASET VALIDATION")
    
    if clinical_df is None or mutation_df is None:
        log("Cannot cross-validate - missing dataset(s)", "ERROR")
        return
    
    # Extract patient IDs
    clinical_ids = set(clinical_df["cases.submitter_id"].dropna().unique())
    mutation_ids = set(mutation_df["patient_id"].dropna().unique())
    
    log(f"📇 PATIENT ID OVERLAP:")
    log(f"  Clinical IDs: {len(clinical_ids)}")
    log(f"  Mutation IDs: {len(mutation_ids)}")
    
    # Overlap analysis
    overlap = clinical_ids & mutation_ids
    clinical_only = clinical_ids - mutation_ids
    mutation_only = mutation_ids - clinical_ids
    
    log(f"\n  Overlap: {len(overlap)} patients")
    log(f"  Clinical only: {len(clinical_only)} patients")
    log(f"  Mutation only: {len(mutation_only)} patients")
    
    if len(clinical_only) > 0:
        log(f"\n  ⚠️  {len(clinical_only)} patients in clinical but not in mutations", "WARN")
        log(f"     Sample IDs: {list(clinical_only)[:5]}")
    
    if len(mutation_only) > 0:
        log(f"\n  ⚠️  {len(mutation_only)} patients in mutations but not in clinical", "WARN")
        log(f"     Sample IDs: {list(mutation_only)[:5]}")
    
    # Estimate data loss from inner join
    log(f"\n💥 EXPECTED DATA LOSS (inner join):")
    join_size = len(overlap)
    clinical_loss = len(clinical_ids) - join_size
    mutation_loss = len(mutation_ids) - join_size
    
    log(f"  Merged dataset will have: {join_size} patients")
    log(f"  Lost from clinical: {clinical_loss} ({100*clinical_loss/len(clinical_ids):.1f}%)")
    log(f"  Lost from mutation: {mutation_loss} ({100*mutation_loss/len(mutation_ids):.1f}%)")
    
    return {
        "overlap": len(overlap),
        "clinical_only": len(clinical_only),
        "mutation_only": len(mutation_only)
    }


def generate_visualizations(clinical_df, mutation_df, mut_counts, patient_burdens):
    """Generate diagnostic plots."""
    section_header("PHASE 4: VISUALIZATION GENERATION")
    
    log("Generating diagnostic plots...")
    
    # 1. Clinical structure
    if clinical_df is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Vital status
        vital = clinical_df["demographic.vital_status"].value_counts()
        axes[0, 0].bar(vital.index, vital.values, color=['green', 'red'])
        axes[0, 0].set_title("Vital Status Distribution")
        axes[0, 0].set_ylabel("Count")
        
        # Age distribution
        age = pd.to_numeric(clinical_df["diagnoses.age_at_diagnosis"], errors='coerce')
        if age.median() > 2000:
            age = age / 365.25
        axes[0, 1].hist(age.dropna(), bins=30, color='skyblue', edgecolor='black')
        axes[0, 1].set_title("Age at Diagnosis (years)")
        axes[0, 1].set_xlabel("Age")
        axes[0, 1].set_ylabel("Count")
        
        # Stage distribution
        stage = clinical_df["diagnoses.ajcc_pathologic_stage"].value_counts().head(10)
        axes[1, 0].barh(range(len(stage)), stage.values)
        axes[1, 0].set_yticks(range(len(stage)))
        axes[1, 0].set_yticklabels(stage.index)
        axes[1, 0].set_title("Top 10 Tumor Stages")
        axes[1, 0].set_xlabel("Count")
        
        # Missing data heatmap (sample)
        key_cols = ["demographic.vital_status", "diagnoses.age_at_diagnosis", 
                    "diagnoses.ajcc_pathologic_stage", "diagnoses.primary_diagnosis"]
        missing_matrix = clinical_df[key_cols].isna().astype(int)
        axes[1, 1].imshow(missing_matrix.T, aspect='auto', cmap='RdYlGn_r', interpolation='nearest')
        axes[1, 1].set_title("Missing Data Pattern (Key Columns)")
        axes[1, 1].set_yticks(range(len(key_cols)))
        axes[1, 1].set_yticklabels(key_cols, fontsize=8)
        axes[1, 1].set_xlabel("Patient Index")
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "clinical_structure.png"), dpi=150)
        plt.close()
        log("  ✓ Saved clinical_structure.png")
    
    # 2. Mutation sparsity
    if mutation_df is not None and mut_counts is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Mutation frequency distribution (log scale)
        axes[0, 0].hist(np.log10(mut_counts + 1), bins=50, color='coral', edgecolor='black')
        axes[0, 0].set_title("Mutation Frequency Distribution (log10)")
        axes[0, 0].set_xlabel("log10(mutations + 1)")
        axes[0, 0].set_ylabel("Number of genes")
        
        # Top 30 genes
        top30 = mut_counts.head(30)
        axes[0, 1].barh(range(len(top30)), top30.values)
        axes[0, 1].set_yticks(range(len(top30)))
        axes[0, 1].set_yticklabels(top30.index, fontsize=8)
        axes[0, 1].set_title("Top 30 Most Mutated Genes")
        axes[0, 1].set_xlabel("Mutation Count")
        
        # Patient mutation burden
        axes[1, 0].hist(patient_burdens, bins=50, color='purple', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title("Patient Mutation Burden")
        axes[1, 0].set_xlabel("Total Mutations per Patient")
        axes[1, 0].set_ylabel("Count")
        
        # Sparsity visualization (sample)
        gene_cols = [c for c in mutation_df.columns if c != "patient_id"]
        sample_genes = np.random.choice(gene_cols, min(100, len(gene_cols)), replace=False)
        sample_matrix = mutation_df[sample_genes].values[:100]  # First 100 patients
        axes[1, 1].imshow(sample_matrix.T, aspect='auto', cmap='Greys', interpolation='nearest')
        axes[1, 1].set_title("Mutation Sparsity (100 genes × 100 patients)")
        axes[1, 1].set_xlabel("Patient Index")
        axes[1, 1].set_ylabel("Gene Index")
        
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "mutation_sparsity.png"), dpi=150)
        plt.close()
        log("  ✓ Saved mutation_sparsity.png")


def save_quality_report(clinical_df, mutation_df, overlap_stats):
    """Save structured quality flags."""
    section_header("PHASE 5: QUALITY REPORT GENERATION")
    
    # Helper to convert numpy/pandas types to native Python
    def make_serializable(obj):
        """Convert numpy/pandas types to native Python types for JSON."""
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        else:
            return obj
    
    flags = {
        "timestamp": datetime.now().isoformat(),
        "clinical": {},
        "mutation": {},
        "overlap": make_serializable(overlap_stats) if overlap_stats else {},
        "warnings": [],
        "errors": []
    }
    
    # Clinical flags
    if clinical_df is not None:
        flags["clinical"] = {
            "n_patients": int(len(clinical_df)),
            "n_columns": int(len(clinical_df.columns)),
            "vital_status_available": bool("demographic.vital_status" in clinical_df.columns),
            "age_available": bool("diagnoses.age_at_diagnosis" in clinical_df.columns),
            "stage_available": bool("diagnoses.ajcc_pathologic_stage" in clinical_df.columns)
        }
    else:
        flags["errors"].append("Clinical data not loaded")
    
    # Mutation flags
    if mutation_df is not None:
        gene_cols = [c for c in mutation_df.columns if c != "patient_id"]
        mut_counts = mutation_df[gene_cols].sum()
        flags["mutation"] = {
            "n_patients": int(len(mutation_df)),
            "n_genes": int(len(gene_cols)),
            "sparsity_pct": float(100 * (1 - mut_counts.sum() / (len(mutation_df) * len(gene_cols)))),
            "genes_with_zero_muts": int((mut_counts == 0).sum()),
            "genes_with_one_mut": int((mut_counts == 1).sum())
        }
    else:
        flags["errors"].append("Mutation data not loaded")
    
    # Save
    report_path = os.path.join(REPORT_DIR, "data_quality_flags.json")
    with open(report_path, 'w') as f:
        json.dump(flags, f, indent=2)
    
    log(f"✓ Saved quality flags: {report_path}")
    
    return flags


def save_text_report():
    """Save full text report."""
    report_path = os.path.join(REPORT_DIR, "data_archaeology_report.txt")
    with open(report_path, 'w') as f:
        f.write("\n".join(REPORT))
    log(f"\n✓ Full report saved: {report_path}")


def main():
    """Main execution."""
    section_header("DATA ARCHAEOLOGY PIPELINE")
    log("Starting forensic inspection of datasets...")
    
    # Phase 1: Clinical
    clinical_df = inspect_clinical_raw()
    
    # Phase 2: Mutations
    mutation_df, mut_counts, patient_burdens = None, None, None
    if os.path.exists(MUTATION_FILE):
        result = inspect_mutation_matrix()
        if result:
            mutation_df, mut_counts, patient_burdens = result
    else:
        log(f"\n⚠️  Mutation matrix not found: {MUTATION_FILE}", "WARN")
        log("   You'll need to generate it from MAF files first.", "INFO")
    
    # Phase 3: Cross-validation
    overlap_stats = cross_validate_datasets(clinical_df, mutation_df)
    
    # Phase 4: Visualizations
    generate_visualizations(clinical_df, mutation_df, mut_counts, patient_burdens)
    
    # Phase 5: Quality report
    flags = save_quality_report(clinical_df, mutation_df, overlap_stats)
    
    # Save text report
    save_text_report()
    
    section_header("ARCHAEOLOGY COMPLETE")
    log("All reports saved to reports/")
    log("All figures saved to figures/archaeology/")
    log("\n🎯 NEXT STEPS:")
    log("  1. Review reports/data_archaeology_report.txt")
    log("  2. Check figures/archaeology/ for visual diagnostics")
    log("  3. Review reports/data_quality_flags.json for automated flags")
    log("  4. If mutation matrix missing, run MAF processing pipeline")
    log("  5. Proceed to data cleaning pipeline (01_data_cleaning.py)")


if __name__ == "__main__":
    main()