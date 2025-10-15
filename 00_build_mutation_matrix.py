#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
00b_build_mutation_matrix_v2.py

ENHANCED MUTATION MATRIX CONSTRUCTION
======================================
Hybrid approach: Simple logic from original + Comprehensive QC

IMPROVEMENTS OVER ORIGINAL:
  1. Whitelist-based variant filtering (more rigorous)
  2. Patient ID validation
  3. Comprehensive error handling
  4. Detailed statistics & reporting
  5. QC visualizations
  6. Processing progress tracking

INPUTS:
  - data/maf_files/*.maf.gz (992 files expected)

OUTPUTS:
  - results/master_mutation_matrix.csv
  - reports/mutation_processing_log.txt
  - reports/mutation_qc_report.json
  - figures/maf_qc/mutation_matrix_qc.png
"""

import os
import json
import gzip
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
DATA_DIR = "data"
MAF_DIR = os.path.join(DATA_DIR, "maf_files")
RESULTS_DIR = "results"
REPORT_DIR = "reports"
FIG_DIR = os.path.join("figures", "maf_qc")

OUTPUT_FILE = os.path.join(RESULTS_DIR, "master_mutation_matrix.csv")
LOG_FILE = os.path.join(REPORT_DIR, "mutation_processing_log.txt")
QC_REPORT_FILE = os.path.join(REPORT_DIR, "mutation_qc_report.json")

# Create directories
for d in [RESULTS_DIR, REPORT_DIR, FIG_DIR]:
    os.makedirs(d, exist_ok=True)

# MAF columns we need
REQUIRED_COLS = ['Hugo_Symbol', 'Variant_Classification', 'Tumor_Sample_Barcode']

# Impactful variant types (whitelist approach - safer than blacklist)
IMPACTFUL_VARIANTS = {
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Nonstop_Mutation",
    "Splice_Site",
    "Translation_Start_Site"
}

# Logging
LOG = []

def log(msg, level="INFO"):
    """Log with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] [{level}] {msg}"
    print(formatted)
    LOG.append(formatted)

def section_header(title):
    """Print section header."""
    border = "=" * 80
    log(f"\n{border}")
    log(f"  {title}")
    log(f"{border}\n")

# ============================================================================
# STEP 1: DISCOVER MAF FILES
# ============================================================================

def discover_maf_files():
    """Find all MAF files in data/maf_files/."""
    section_header("STEP 1: DISCOVER MAF FILES")
    
    log(f"Searching for MAF files in: {MAF_DIR}")
    
    # Support both .maf and .maf.gz
    patterns = ['*.maf', '*.maf.gz']
    maf_files = []
    
    for pattern in patterns:
        import glob
        files = glob.glob(os.path.join(MAF_DIR, pattern))
        maf_files.extend(files)
    
    if len(maf_files) == 0:
        log(f"✗ ERROR: No MAF files found in {MAF_DIR}", "ERROR")
        log(f"  Expected patterns: *.maf or *.maf.gz", "INFO")
        return []
    
    log(f"✓ Found {len(maf_files)} MAF files")
    
    # Calculate total size
    total_size_mb = sum(os.path.getsize(f) for f in maf_files) / 1e6
    log(f"  Total size: {total_size_mb:.1f} MB")
    
    return maf_files

# ============================================================================
# STEP 2: READ AND CONSOLIDATE MAF FILES
# ============================================================================

def read_single_maf(filepath):
    """
    Read a single MAF file (handles .maf and .maf.gz).
    Returns DataFrame or None if error.
    """
    try:
        # Determine if gzipped
        is_gzipped = filepath.endswith('.gz')
        
        if is_gzipped:
            with gzip.open(filepath, 'rt') as f:
                # Skip comment lines
                lines = [line for line in f if not line.startswith('#')]
                from io import StringIO
                df = pd.read_csv(StringIO(''.join(lines)), sep='\t', 
                                usecols=REQUIRED_COLS, low_memory=False)
        else:
            df = pd.read_csv(filepath, sep='\t', comment='#', 
                            usecols=REQUIRED_COLS, low_memory=False)
        
        return df
    
    except Exception as e:
        return None

def consolidate_maf_files(maf_files):
    """Read all MAF files and consolidate into single DataFrame."""
    section_header("STEP 2: CONSOLIDATE MAF FILES")
    
    all_mutations = []
    stats = {
        "processed": 0,
        "failed": 0,
        "empty": 0,
        "total_rows": 0,
        "variant_types": Counter()
    }
    
    log(f"Processing {len(maf_files)} MAF files...")
    
    for filepath in tqdm(maf_files, desc="📖 Reading MAF files"):
        filename = os.path.basename(filepath)
        
        # Read file
        df = read_single_maf(filepath)
        
        if df is None:
            tqdm.write(f"  ⚠️  Failed to read: {filename}")
            stats["failed"] += 1
            continue
        
        if len(df) == 0:
            tqdm.write(f"  ⚠️  Empty file: {filename}")
            stats["empty"] += 1
            continue
        
        # Check for required columns
        missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing_cols:
            tqdm.write(f"  ⚠️  Missing columns in {filename}: {missing_cols}")
            stats["failed"] += 1
            continue
        
        # Count variant types
        for var_type in df['Variant_Classification']:
            stats["variant_types"][var_type] += 1
        
        all_mutations.append(df)
        stats["processed"] += 1
        stats["total_rows"] += len(df)
    
    if len(all_mutations) == 0:
        log("✗ ERROR: No MAF files successfully processed!", "ERROR")
        return None, stats
    
    # Concatenate all
    master_df = pd.concat(all_mutations, ignore_index=True)
    
    log(f"\n📊 CONSOLIDATION SUMMARY:")
    log(f"  Files processed: {stats['processed']} / {len(maf_files)}")
    log(f"  Files failed: {stats['failed']}")
    log(f"  Files empty: {stats['empty']}")
    log(f"  Total mutations: {len(master_df):,}")
    
    log(f"\n🧬 VARIANT TYPE BREAKDOWN (top 10):")
    for var_type, count in stats['variant_types'].most_common(10):
        pct = 100 * count / stats['total_rows']
        log(f"  {var_type}: {count:,} ({pct:.1f}%)")
    
    return master_df, stats

# ============================================================================
# STEP 3: FILTER & VALIDATE
# ============================================================================

def filter_and_validate(master_df):
    """Filter mutations and validate data quality."""
    section_header("STEP 3: FILTER & VALIDATE")
    
    n_original = len(master_df)
    log(f"Starting with {n_original:,} mutations")
    
    # === FILTER 1: Impactful Variants Only ===
    log(f"\n🎯 FILTER 1: Impactful variants only")
    log(f"  Keeping: {', '.join(sorted(IMPACTFUL_VARIANTS))}")
    
    impactful = master_df[
        master_df['Variant_Classification'].isin(IMPACTFUL_VARIANTS)
    ].copy()
    
    n_filtered = len(impactful)
    pct_kept = 100 * n_filtered / n_original
    log(f"  Kept: {n_filtered:,} / {n_original:,} ({pct_kept:.1f}%)")
    log(f"  Removed: {n_original - n_filtered:,} non-impactful variants")
    
    # === FILTER 2: Extract & Validate Patient IDs ===
    log(f"\n🔍 FILTER 2: Patient ID extraction & validation")
    
    # Extract first 12 characters (TCGA-XX-XXXX)
    impactful['patient_id'] = impactful['Tumor_Sample_Barcode'].str.slice(0, 12)
    
    # Validate format: TCGA-XX-XXXX
    valid_pattern = impactful['patient_id'].str.match(r'^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$')
    n_invalid = (~valid_pattern).sum()
    
    if n_invalid > 0:
        log(f"  ⚠️  {n_invalid} invalid patient IDs detected", "WARN")
        # Show samples
        invalid_samples = impactful.loc[~valid_pattern, 'Tumor_Sample_Barcode'].head(5)
        for sample in invalid_samples:
            log(f"    Example: {sample}")
        
        # Remove invalid
        impactful = impactful[valid_pattern].copy()
        log(f"  Removed {n_invalid} rows with invalid IDs")
    else:
        log(f"  ✓ All patient IDs valid")
    
    log(f"  Unique patients: {impactful['patient_id'].nunique()}")
    
    # === FILTER 3: Gene Name Validation ===
    log(f"\n🧬 FILTER 3: Gene name validation")
    
    # More permissive gene name validation
    valid_genes = impactful['Hugo_Symbol'].str.match(r'^[A-Za-z0-9\-\.]+$', na=False)
    n_invalid_genes = (~valid_genes).sum()
    
    if n_invalid_genes > 0:
        log(f"  ⚠️  {n_invalid_genes} invalid gene names", "WARN")
        # Show samples
        invalid_gene_samples = impactful.loc[~valid_genes, 'Hugo_Symbol'].head(5)
        for gene in invalid_gene_samples:
            log(f"    Example: {gene}")
        
        impactful = impactful[valid_genes].copy()
        log(f"  Removed {n_invalid_genes} rows with invalid gene names")
    else:
        log(f"  ✓ All gene names valid")
    
    log(f"  Unique genes: {impactful['Hugo_Symbol'].nunique()}")
    
    # === Final counts ===
    log(f"\n📊 FILTERING SUMMARY:")
    log(f"  Original: {n_original:,}")
    log(f"  Final: {len(impactful):,}")
    log(f"  Removed: {n_original - len(impactful):,} ({100*(n_original-len(impactful))/n_original:.1f}%)")
    
    return impactful

# ============================================================================
# STEP 4: CREATE BINARY MATRIX
# ============================================================================

def create_mutation_matrix(filtered_df):
    """Create patient × gene binary mutation matrix."""
    section_header("STEP 4: CREATE BINARY MATRIX")
    
    log("Preparing data for matrix creation...")
    
    # Keep only patient_id and gene
    mutation_pairs = filtered_df[['patient_id', 'Hugo_Symbol']].copy()
    
    # Remove duplicates (same patient, same gene)
    n_before_dedup = len(mutation_pairs)
    mutation_pairs = mutation_pairs.drop_duplicates()
    n_duplicates = n_before_dedup - len(mutation_pairs)
    
    if n_duplicates > 0:
        log(f"  Removed {n_duplicates:,} duplicate mutations (same patient + gene)")
    
    log(f"  Unique mutation events: {len(mutation_pairs):,}")
    log(f"  Unique patients: {mutation_pairs['patient_id'].nunique()}")
    log(f"  Unique genes: {mutation_pairs['Hugo_Symbol'].nunique()}")
    
    # Create binary indicator
    mutation_pairs['mutated'] = 1
    
    # Pivot to wide format
    log("\n🔄 Pivoting to matrix format...")
    matrix = mutation_pairs.pivot_table(
        index='patient_id',
        columns='Hugo_Symbol',
        values='mutated',
        fill_value=0,
        aggfunc='max'  # In case of any remaining duplicates
    )
    
    # Reset index to make patient_id a column
    matrix = matrix.reset_index()
    
    log(f"✓ Matrix created: {matrix.shape[0]} patients × {matrix.shape[1]-1} genes")
    
    # Optimize data types
    log("\n💾 Optimizing memory usage...")
    gene_cols = [c for c in matrix.columns if c != 'patient_id']
    
    memory_before = matrix.memory_usage(deep=True).sum() / 1e6
    
    for col in gene_cols:
        matrix[col] = matrix[col].astype(np.int8)
    
    memory_after = matrix.memory_usage(deep=True).sum() / 1e6
    log(f"  Before optimization: {memory_before:.2f} MB")
    log(f"  After optimization: {memory_after:.2f} MB")
    log(f"  Saved: {memory_before - memory_after:.2f} MB ({100*(memory_before-memory_after)/memory_before:.1f}%)")
    
    return matrix

# ============================================================================
# STEP 5: QUALITY CONTROL
# ============================================================================

def run_quality_control(matrix):
    """Perform quality checks on mutation matrix."""
    section_header("STEP 5: QUALITY CONTROL")
    
    gene_cols = [c for c in matrix.columns if c != 'patient_id']
    gene_sums = matrix[gene_cols].sum()
    patient_burdens = matrix[gene_cols].sum(axis=1)
    
    qc_results = {}
    
    # === QC 1: All-zero genes ===
    log("🔍 QC Check 1: All-zero genes")
    zero_genes = gene_sums[gene_sums == 0].index.tolist()
    
    if zero_genes:
        log(f"  ⚠️  {len(zero_genes)} genes with ZERO mutations", "WARN")
        log(f"     (These should be filtered out)")
    else:
        log(f"  ✓ No all-zero genes")
    
    qc_results['zero_genes'] = zero_genes
    
    # === QC 2: All-one genes ===
    log(f"\n🔍 QC Check 2: All-one genes")
    all_one_genes = gene_sums[gene_sums == len(matrix)].index.tolist()
    
    if all_one_genes:
        log(f"  ⚠️  {len(all_one_genes)} genes mutated in ALL patients", "WARN")
    else:
        log(f"  ✓ No all-one genes")
    
    qc_results['all_one_genes'] = all_one_genes
    
    # === QC 3: Mutation frequency statistics ===
    log(f"\n📊 QC Check 3: Mutation frequency distribution")
    log(f"  Mean mutations per gene: {gene_sums.mean():.2f}")
    log(f"  Median mutations per gene: {gene_sums.median():.0f}")
    log(f"  Min: {gene_sums.min():.0f}")
    log(f"  Max: {gene_sums.max():.0f}")
    
    # Rare mutation breakdown
    log(f"\n  📉 Rare mutation distribution:")
    for threshold in [1, 2, 5, 10, 20, 50]:
        n_rare = (gene_sums <= threshold).sum()
        pct = 100 * n_rare / len(gene_sums)
        log(f"    Genes with ≤{threshold:2d} mutations: {n_rare:5d} ({pct:.1f}%)")
    
    qc_results['gene_sums'] = gene_sums
    
    # === QC 4: Top mutated genes ===
    log(f"\n🔝 QC Check 4: Top 20 mutated genes")
    top20 = gene_sums.sort_values(ascending=False).head(20)
    for i, (gene, count) in enumerate(top20.items(), 1):
        pct = 100 * count / len(matrix)
        log(f"  {i:2d}. {gene}: {int(count)} ({pct:.1f}%)")
    
    # === QC 5: Patient mutation burden ===
    log(f"\n👤 QC Check 5: Patient mutation burden")
    log(f"  Mean: {patient_burdens.mean():.2f}")
    log(f"  Median: {patient_burdens.median():.0f}")
    log(f"  Min: {patient_burdens.min():.0f}")
    log(f"  Max: {patient_burdens.max():.0f}")
    log(f"  Std: {patient_burdens.std():.2f}")
    
    # Identify outliers (>5 std from mean)
    mean_burden = patient_burdens.mean()
    std_burden = patient_burdens.std()
    outlier_threshold = mean_burden + 5 * std_burden
    outliers = matrix[patient_burdens > outlier_threshold]['patient_id'].tolist()
    
    if outliers:
        log(f"\n  ⚠️  {len(outliers)} HYPERMUTATOR patients detected (>5σ)", "WARN")
        for patient in outliers:
            burden = patient_burdens[matrix['patient_id'] == patient].values[0]
            z_score = (burden - mean_burden) / std_burden
            log(f"    {patient}: {burden:.0f} mutations (Z={z_score:.2f})")
        log(f"  These patients may need special handling in analysis.")
    else:
        log(f"  ✓ No extreme outliers detected")
    
    qc_results['outlier_patients'] = outliers
    qc_results['patient_burdens'] = patient_burdens
    
    # === QC 6: Matrix sparsity ===
    log(f"\n🕳️  QC Check 6: Matrix sparsity")
    total_cells = len(matrix) * len(gene_cols)
    mutated_cells = gene_sums.sum()
    sparsity = 100 * (1 - mutated_cells / total_cells)
    
    log(f"  Total cells: {total_cells:,}")
    log(f"  Mutated cells: {int(mutated_cells):,}")
    log(f"  Sparsity: {sparsity:.2f}%")
    log(f"  (Expected ~99% for cancer genomics)")
    
    qc_results['sparsity_pct'] = sparsity
    
    return qc_results

# ============================================================================
# STEP 6: GENERATE QC PLOTS
# ============================================================================

def generate_qc_plots(matrix, qc_results):
    """Generate QC visualization plots."""
    section_header("STEP 6: GENERATE QC VISUALIZATIONS")
    
    gene_cols = [c for c in matrix.columns if c != 'patient_id']
    gene_sums = qc_results['gene_sums']
    patient_burdens = qc_results['patient_burdens']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mutation frequency distribution (log scale)
    axes[0, 0].hist(np.log10(gene_sums + 1), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('log10(Mutation Count + 1)', fontsize=11)
    axes[0, 0].set_ylabel('Number of Genes', fontsize=11)
    axes[0, 0].set_title('Gene Mutation Frequency Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Top 30 genes
    top30 = gene_sums.sort_values(ascending=False).head(30)
    y_pos = np.arange(len(top30))
    axes[0, 1].barh(y_pos, top30.values, color='coral', edgecolor='black', alpha=0.8)
    axes[0, 1].set_yticks(y_pos)
    axes[0, 1].set_yticklabels(top30.index, fontsize=9)
    axes[0, 1].set_xlabel('Mutation Count', fontsize=11)
    axes[0, 1].set_title('Top 30 Most Mutated Genes', fontsize=12, fontweight='bold')
    axes[0, 1].invert_yaxis()
    axes[0, 1].grid(axis='x', alpha=0.3)
    
    # Plot 3: Patient mutation burden
    axes[1, 0].hist(patient_burdens, bins=50, color='mediumseagreen', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(patient_burdens.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {patient_burdens.mean():.1f}')
    axes[1, 0].axvline(patient_burdens.median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {patient_burdens.median():.1f}')
    axes[1, 0].set_xlabel('Total Mutations per Patient', fontsize=11)
    axes[1, 0].set_ylabel('Number of Patients', fontsize=11)
    axes[1, 0].set_title('Patient Mutation Burden Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Sparsity heatmap (sample)
    n_sample_genes = min(100, len(gene_cols))
    n_sample_patients = min(100, len(matrix))
    np.random.seed(42)
    sample_genes = np.random.choice(gene_cols, n_sample_genes, replace=False)
    sample_matrix = matrix[sample_genes].iloc[:n_sample_patients].T.values
    
    im = axes[1, 1].imshow(sample_matrix, aspect='auto', cmap='binary', interpolation='nearest')
    axes[1, 1].set_xlabel('Patient Index', fontsize=11)
    axes[1, 1].set_ylabel('Gene Index', fontsize=11)
    axes[1, 1].set_title(f'Mutation Matrix Sparsity\n({n_sample_genes} genes × {n_sample_patients} patients)', 
                         fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "mutation_matrix_qc.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log(f"✓ QC plots saved: {path}")

# ============================================================================
# STEP 7: SAVE OUTPUTS
# ============================================================================

def save_outputs(matrix, processing_stats, qc_results):
    """Save all outputs: matrix, logs, reports."""
    section_header("STEP 7: SAVE OUTPUTS")
    
    # === Output 1: Mutation Matrix ===
    log(f"💾 Saving mutation matrix...")
    matrix.to_csv(OUTPUT_FILE, index=False)
    file_size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
    log(f"  ✓ Saved: {OUTPUT_FILE}")
    log(f"  Size: {file_size_mb:.2f} MB")
    
    # === Output 2: Processing Log ===
    log(f"\n📝 Saving processing log...")
    with open(LOG_FILE, 'w') as f:
        f.write('\n'.join(LOG))
    log(f"  ✓ Saved: {LOG_FILE}")
    
    # === Output 3: QC Report (JSON) ===
    log(f"\n📊 Saving QC report...")
    
    gene_cols = [c for c in matrix.columns if c != 'patient_id']
    gene_sums = qc_results['gene_sums']
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "processing": {
            "maf_files_processed": processing_stats['processed'],
            "maf_files_failed": processing_stats['failed'],
            "maf_files_empty": processing_stats['empty'],
            "total_mutations_raw": processing_stats['total_rows']
        },
        "matrix": {
            "n_patients": int(len(matrix)),
            "n_genes": int(len(gene_cols)),
            "total_mutations": int(gene_sums.sum()),
            "sparsity_pct": float(qc_results['sparsity_pct'])
        },
        "gene_stats": {
            "mean_mutations_per_gene": float(gene_sums.mean()),
            "median_mutations_per_gene": float(gene_sums.median()),
            "max_mutations": int(gene_sums.max()),
            "genes_with_zero_muts": len(qc_results['zero_genes']),
            "genes_with_one_mut": int((gene_sums == 1).sum())
        },
        "patient_stats": {
            "mean_mutations_per_patient": float(qc_results['patient_burdens'].mean()),
            "median_mutations_per_patient": float(qc_results['patient_burdens'].median()),
            "max_mutations": int(qc_results['patient_burdens'].max()),
            "hypermutator_patients": qc_results['outlier_patients']
        },
        "top_10_genes": {
            gene: int(count) 
            for gene, count in gene_sums.sort_values(ascending=False).head(10).items()
        },
        "variant_type_breakdown": {
            var_type: int(count) 
            for var_type, count in processing_stats['variant_types'].most_common(10)
        }
    }
    
    with open(QC_REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
    
    log(f"  ✓ Saved: {QC_REPORT_FILE}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""
    section_header("MUTATION MATRIX CONSTRUCTION PIPELINE V2")
    log("🚀 Starting enhanced mutation matrix construction...")
    log(f"Source directory: {MAF_DIR}")
    log(f"Output file: {OUTPUT_FILE}")
    
    # Step 1: Discover MAF files
    maf_files = discover_maf_files()
    if len(maf_files) == 0:
        log("\n❌ No MAF files found. Pipeline stopped.", "ERROR")
        return
    
    # Step 2: Consolidate MAF files
    master_df, processing_stats = consolidate_maf_files(maf_files)
    if master_df is None:
        log("\n❌ MAF consolidation failed. Pipeline stopped.", "ERROR")
        return
    
    # Step 3: Filter and validate
    filtered_df = filter_and_validate(master_df)
    
    # Step 4: Create binary matrix
    matrix = create_mutation_matrix(filtered_df)
    
    # Step 5: Quality control
    qc_results = run_quality_control(matrix)
    
    # Step 6: Generate QC plots
    generate_qc_plots(matrix, qc_results)
    
    # Step 7: Save outputs
    save_outputs(matrix, processing_stats, qc_results)
    
    # Final summary
    section_header("✅ PIPELINE COMPLETE")
    log("All outputs saved successfully:")
    log(f"  📁 Matrix: {OUTPUT_FILE}")
    log(f"  📁 Log: {LOG_FILE}")
    log(f"  📁 QC Report: {QC_REPORT_FILE}")
    log(f"  📁 QC Plots: {FIG_DIR}/")
    
    log(f"\n📊 FINAL STATISTICS:")
    log(f"  Patients: {len(matrix)}")
    log(f"  Genes: {len([c for c in matrix.columns if c != 'patient_id'])}")
    log(f"  Total mutations: {int(qc_results['gene_sums'].sum())}")
    log(f"  Sparsity: {qc_results['sparsity_pct']:.2f}%")
    
    if qc_results['outlier_patients']:
        log(f"\n⚠️  WARNING: {len(qc_results['outlier_patients'])} hypermutator patient(s) detected", "WARN")
        log(f"  These should be reviewed during cleaning phase.")
    
    log(f"\n🎯 NEXT STEPS:")
    log(f"  1. Review QC report: {QC_REPORT_FILE}")
    log(f"  2. Check QC plots: {FIG_DIR}/mutation_matrix_qc.png")
    log(f"  3. Run archaeology: python 00_data_archaeology.py")
    log(f"  4. Proceed to cleaning: python 01_data_cleaning.py")
    
    log("\n🎉 Mutation matrix construction complete!")


if __name__ == "__main__":
    main()