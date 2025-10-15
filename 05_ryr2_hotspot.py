#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05_ryr2_mutation_hotspot_analysis.py

MUTATION PATTERN DEEP DIVE - RYR2
==================================
Analyze mutation patterns in RYR2 from MAF files:

1. Mutation positions (scattered vs clustered?)
2. Mutation types (missense, nonsense, frameshift)
3. Protein domain analysis
4. Hotspot detection (recurrent positions)
5. Mutation signature association (COSMIC signatures)
6. Comparison with COSMIC/cBioPortal data

HYPOTHESIS: 
  If RYR2 is a passenger → SCATTERED mutations across gene
  If RYR2 is a driver → CLUSTERED mutations at hotspots
"""

import os
import gzip
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
MAF_DIR = os.path.join(DATA_DIR, "maf_files")
RESULTS_DIR = "results"
FIG_DIR = os.path.join("figures", "deep_validation", "ryr2")

# Create directories
os.makedirs(FIG_DIR, exist_ok=True)

# Logging
LOG = []

def log(msg, level="INFO"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted = f"[{timestamp}] [{level}] {msg}"
    print(formatted)
    LOG.append(formatted)

def section_header(title):
    border = "=" * 80
    log(f"\n{border}")
    log(f"  {title}")
    log(f"{border}\n")

# ============================================================================
# PHASE 1: EXTRACT RYR2 MUTATIONS FROM MAF FILES
# ============================================================================

def extract_ryr2_mutations():
    section_header("PHASE 1: EXTRACT RYR2 MUTATIONS FROM MAF FILES")
    
    import glob
    maf_files = glob.glob(os.path.join(MAF_DIR, "*.maf.gz"))
    
    if len(maf_files) == 0:
        maf_files = glob.glob(os.path.join(MAF_DIR, "*.maf"))
    
    log(f"Found {len(maf_files)} MAF files")
    
    ryr2_mutations = []
    
    for maf_file in maf_files:
        try:
            # Read MAF
            if maf_file.endswith('.gz'):
                with gzip.open(maf_file, 'rt') as f:
                    lines = [line for line in f if not line.startswith('#')]
                    from io import StringIO
                    df = pd.read_csv(StringIO(''.join(lines)), sep='\t', low_memory=False)
            else:
                df = pd.read_csv(maf_file, sep='\t', comment='#', low_memory=False)
            
            # Filter for RYR2
            ryr2_df = df[df['Hugo_Symbol'] == 'RYR2'].copy()
            
            if len(ryr2_df) > 0:
                ryr2_mutations.append(ryr2_df)
        
        except Exception as e:
            log(f"Error reading {os.path.basename(maf_file)}: {e}", "WARN")
            continue
    
    if len(ryr2_mutations) == 0:
        log("No RYR2 mutations found in MAF files!", "ERROR")
        return None
    
    # Combine all RYR2 mutations
    all_ryr2 = pd.concat(ryr2_mutations, ignore_index=True)
    
    log(f"Total RYR2 mutations: {len(all_ryr2)}")
    
    # Extract patient IDs
    all_ryr2['patient_id'] = all_ryr2['Tumor_Sample_Barcode'].str.slice(0, 12)
    
    log(f"Unique patients with RYR2 mutations: {all_ryr2['patient_id'].nunique()}")
    
    return all_ryr2

# ============================================================================
# PHASE 2: MUTATION TYPE ANALYSIS
# ============================================================================

def analyze_mutation_types(ryr2_muts):
    section_header("PHASE 2: MUTATION TYPE DISTRIBUTION")
    
    # Variant classification
    var_types = ryr2_muts['Variant_Classification'].value_counts()
    
    log("Variant classification distribution:")
    for var_type, count in var_types.items():
        pct = 100 * count / len(ryr2_muts)
        log(f"  {var_type}: {count} ({pct:.1f}%)")
    
    # Variant type
    if 'Variant_Type' in ryr2_muts.columns:
        vtype = ryr2_muts['Variant_Type'].value_counts()
        log("\nVariant type distribution:")
        for vt, count in vtype.items():
            pct = 100 * count / len(ryr2_muts)
            log(f"  {vt}: {count} ({pct:.1f}%)")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Variant classification
    var_types_top = var_types.head(10)
    axes[0].barh(range(len(var_types_top)), var_types_top.values, color='steelblue')
    axes[0].set_yticks(range(len(var_types_top)))
    axes[0].set_yticklabels(var_types_top.index, fontsize=9)
    axes[0].set_xlabel('Count', fontsize=11)
    axes[0].set_title('RYR2 Variant Classification', fontsize=12, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(alpha=0.3, axis='x')
    
    # Plot 2: Functional impact
    impactful = ['Missense_Mutation', 'Nonsense_Mutation', 'Frame_Shift_Del', 
                 'Frame_Shift_Ins', 'In_Frame_Del', 'In_Frame_Ins', 'Splice_Site']
    
    ryr2_muts['is_impactful'] = ryr2_muts['Variant_Classification'].isin(impactful)
    impact_dist = ryr2_muts['is_impactful'].value_counts()
    
    axes[1].pie(impact_dist.values, labels=['Impactful', 'Non-impactful'], 
                autopct='%1.1f%%', startangle=90, colors=['coral', 'lightgray'])
    axes[1].set_title('RYR2 Functional Impact', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "ryr2_mutation_types.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log(f"\n✓ Saved: {path}")
    
    return ryr2_muts

# ============================================================================
# PHASE 3: HOTSPOT ANALYSIS
# ============================================================================

def hotspot_analysis(ryr2_muts):
    section_header("PHASE 3: HOTSPOT ANALYSIS - Position Distribution")
    
    # Extract positions
    if 'Start_Position' in ryr2_muts.columns:
        positions = ryr2_muts['Start_Position'].values
    elif 'Start_position' in ryr2_muts.columns:
        positions = ryr2_muts['Start_position'].values
    else:
        log("Position column not found!", "ERROR")
        return
    
    # Count recurrent positions
    pos_counts = Counter(positions)
    recurrent = {pos: count for pos, count in pos_counts.items() if count > 1}
    
    log(f"Total mutation positions: {len(pos_counts)}")
    log(f"Recurrent positions (>1 patient): {len(recurrent)}")
    log(f"Unique positions: {len(pos_counts) - len(recurrent)}")
    
    if len(recurrent) > 0:
        log("\nTop 10 recurrent positions:")
        for i, (pos, count) in enumerate(sorted(recurrent.items(), key=lambda x: -x[1])[:10], 1):
            pct = 100 * count / len(ryr2_muts)
            log(f"  {i:2d}. Position {pos}: {count} mutations ({pct:.1f}%)")
    else:
        log("⚠️  NO HOTSPOTS DETECTED - all mutations are unique!")
        log("   This is STRONG evidence for PASSENGER mutations")
    
    # Plot position distribution
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Plot 1: Position histogram
    axes[0].hist(positions, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Genomic Position', fontsize=11)
    axes[0].set_ylabel('Mutation Count', fontsize=11)
    axes[0].set_title('RYR2 Mutation Position Distribution', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3, axis='y')
    
    # Plot 2: Recurrent positions (if any)
    if len(recurrent) > 0:
        top_recurrent = sorted(recurrent.items(), key=lambda x: -x[1])[:20]
        pos_labels = [str(p) for p, c in top_recurrent]
        counts = [c for p, c in top_recurrent]
        
        axes[1].bar(range(len(top_recurrent)), counts, color='coral', edgecolor='black', alpha=0.7)
        axes[1].set_xticks(range(len(top_recurrent)))
        axes[1].set_xticklabels(pos_labels, rotation=45, ha='right', fontsize=8)
        axes[1].set_xlabel('Position', fontsize=11)
        axes[1].set_ylabel('Recurrence Count', fontsize=11)
        axes[1].set_title('Top 20 Recurrent Mutation Positions', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3, axis='y')
    else:
        axes[1].text(0.5, 0.5, 'NO HOTSPOTS DETECTED\n(All mutations are unique)', 
                     ha='center', va='center', fontsize=14, fontweight='bold',
                     transform=axes[1].transAxes)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "ryr2_hotspot_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log(f"\n✓ Saved: {path}")
    
    # Calculate hotspot score
    if len(recurrent) > 0:
        max_recurrence = max(recurrent.values())
        hotspot_score = 100 * sum(recurrent.values()) / len(ryr2_muts)
        log(f"\nHotspot metrics:")
        log(f"  Max recurrence: {max_recurrence} patients at same position")
        log(f"  Hotspot score: {hotspot_score:.1f}% (mutations at recurrent sites)")
    else:
        log(f"\n⚠️  ZERO hotspot score - completely scattered mutations")
        log(f"   STRONG evidence for PASSENGER gene")

# ============================================================================
# PHASE 4: PROTEIN DOMAIN ANALYSIS (if Protein_Change available)
# ============================================================================

def protein_domain_analysis(ryr2_muts):
    section_header("PHASE 4: PROTEIN DOMAIN DISTRIBUTION")
    
    if 'Protein_Change' not in ryr2_muts.columns and 'HGVSp_Short' not in ryr2_muts.columns:
        log("Protein change information not available", "WARN")
        return
    
    # Extract amino acid positions
    if 'HGVSp_Short' in ryr2_muts.columns:
        protein_col = 'HGVSp_Short'
    else:
        protein_col = 'Protein_Change'
    
    # Extract positions from protein change (e.g., p.A123V)
    import re
    
    aa_positions = []
    for pc in ryr2_muts[protein_col].dropna():
        match = re.search(r'(\d+)', str(pc))
        if match:
            aa_positions.append(int(match.group(1)))
    
    if len(aa_positions) == 0:
        log("Could not extract amino acid positions", "WARN")
        return
    
    log(f"Extracted {len(aa_positions)} amino acid positions")
    
    # RYR2 protein domains (approximate)
    # RYR2 is ~4,967 amino acids
    domains = {
        'N-terminal (1-600)': (1, 600),
        'SPRY1 (600-800)': (600, 800),
        'SPRY2 (800-1000)': (800, 1000),
        'SPRY3 (1000-1200)': (1000, 1200),
        'RIH (1200-1400)': (1200, 1400),
        'Central (1400-4000)': (1400, 4000),
        'Transmembrane (4000-4500)': (4000, 4500),
        'C-terminal (4500-4967)': (4500, 4967)
    }
    
    # Count mutations per domain
    domain_counts = {d: 0 for d in domains.keys()}
    
    for pos in aa_positions:
        for domain, (start, end) in domains.items():
            if start <= pos <= end:
                domain_counts[domain] += 1
                break
    
    log("\nMutations per protein domain:")
    for domain, count in domain_counts.items():
        if count > 0:
            pct = 100 * count / len(aa_positions)
            log(f"  {domain}: {count} ({pct:.1f}%)")
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(aa_positions, bins=50, color='coral', edgecolor='black', alpha=0.7)
    
    # Mark domain boundaries
    for domain, (start, end) in domains.items():
        ax.axvline(start, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Amino Acid Position', fontsize=11)
    ax.set_ylabel('Mutation Count', fontsize=11)
    ax.set_title('RYR2 Protein Domain Distribution', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "ryr2_protein_domains.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    
    log(f"\n✓ Saved: {path}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    section_header("RYR2 MUTATION PATTERN DEEP DIVE")
    
    # Extract RYR2 mutations from MAF
    ryr2_muts = extract_ryr2_mutations()
    
    if ryr2_muts is None:
        log("Could not extract RYR2 mutations", "ERROR")
        return
    
    # Analyze mutation types
    ryr2_muts = analyze_mutation_types(ryr2_muts)
    
    # Hotspot analysis
    hotspot_analysis(ryr2_muts)
    
    # Protein domain analysis
    protein_domain_analysis(ryr2_muts)
    
    # Final verdict
    section_header("FINAL VERDICT - MUTATION PATTERN")
    
    log("EXPECTED for PASSENGER gene:")
    log("  ✓ Scattered mutations (no hotspots)")
    log("  ✓ Uniform distribution across protein")
    log("  ✓ High missense:nonsense ratio")
    log("  ✓ No domain-specific clustering")
    log("")
    log("EXPECTED for DRIVER gene:")
    log("  ✓ Clustered mutations (hotspots)")
    log("  ✓ Domain-specific enrichment")
    log("  ✓ Recurrent positions")
    log("  ✓ Functional domain bias")
    log("")
    log("Based on mutation patterns, RYR2 appears to be:")
    log("  [Will be determined after running analysis]")
    
    # Save log
    log_path = os.path.join(FIG_DIR, "ryr2_mutation_pattern_log.txt")
    with open(log_path, 'w') as f:
        f.write('\n'.join(LOG))
    
    log(f"\n✓ Log saved: {log_path}")

if __name__ == "__main__":
    main()