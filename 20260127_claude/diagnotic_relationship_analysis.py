#!/usr/bin/env python3
"""
Diagnostic script: Analyze relationship labeling issues
- Sort pairs by metric values within each relationship category
- Identify potential mislabeled pairs
- Export detailed CSV for inspection
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
HOME = Path.home()
WORK_DIR = HOME / "kinship/Analysis/20251031_wgrs/06_kinship_analysis"
RESULTS_FILE = WORK_DIR / "all_results_combined.csv"
OUTPUT_DIR = WORK_DIR / "diagnostic"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use NFS_36K as reference (or whichever has data)
REFERENCE_MARKER = "NFS_36K"


def main():
    print("=" * 70)
    print("Diagnostic: Relationship Labeling Analysis")
    print("=" * 70)
    
    # Load combined results
    if not RESULTS_FILE.exists():
        print(f"ERROR: Results file not found: {RESULTS_FILE}")
        print("Please run 05_evaluate_results_comprehensive.py first!")
        return
    
    all_df = pd.read_csv(RESULTS_FILE)
    print(f"Loaded {len(all_df):,} rows")
    
    # Filter to reference marker
    df = all_df[all_df['Marker_Set'] == REFERENCE_MARKER].copy()
    print(f"Using marker set: {REFERENCE_MARKER} ({len(df):,} pairs)")
    
    # ============================================================
    # 1. Create comprehensive diagnostic CSV
    # ============================================================
    print("\n[1] Creating comprehensive diagnostic CSV...")
    
    # Select relevant columns
    diagnostic_cols = [
        'Sample1', 'Sample2', 'Family1', 'Family2', 'Member1', 'Member2',
        'Relationship', 'Degree', 'Expected_Kinship', 'Same_Family',
        'IBS', 'IBD', 'Kinship', 'Z0', 'Z1', 'Z2'
    ]
    
    available_cols = [c for c in diagnostic_cols if c in df.columns]
    diagnostic_df = df[available_cols].copy()
    
    # Sort by Relationship, then by IBS descending (to see outliers at top/bottom)
    diagnostic_df = diagnostic_df.sort_values(
        ['Relationship', 'IBS'], 
        ascending=[True, False]
    )
    
    # Save full diagnostic
    output_full = OUTPUT_DIR / "diagnostic_all_pairs_by_relationship.csv"
    diagnostic_df.to_csv(output_full, index=False)
    print(f"  Saved: {output_full}")
    
    # ============================================================
    # 2. Focus on Unrelated pairs - sorted by metrics
    # ============================================================
    print("\n[2] Analyzing Unrelated pairs...")
    
    unrelated = df[df['Relationship'] == 'Unrelated'].copy()
    
    # Sort by IBS descending (highest IBS first - potential issues)
    unrelated_by_ibs = unrelated.sort_values('IBS', ascending=False)
    
    output_unrel = OUTPUT_DIR / "diagnostic_unrelated_sorted_by_IBS.csv"
    unrelated_by_ibs[available_cols].to_csv(output_unrel, index=False)
    print(f"  Saved: {output_unrel}")
    print(f"  Total Unrelated pairs: {len(unrelated):,}")
    
    # Show top suspicious pairs (high IBS for unrelated)
    print("\n  Top 20 Unrelated pairs with HIGHEST IBS (potential mislabeling):")
    print("-" * 100)
    print(f"  {'Sample1':<18} {'Sample2':<18} {'Fam1':>5} {'Fam2':>5} {'IBS':>8} {'IBD':>8} {'Kinship':>8}")
    print("-" * 100)
    
    for _, row in unrelated_by_ibs.head(20).iterrows():
        print(f"  {row['Sample1']:<18} {row['Sample2']:<18} {row['Family1']:>5} {row['Family2']:>5} "
              f"{row['IBS']:>8.4f} {row['IBD']:>8.4f} {row['Kinship']:>8.4f}")
    
    # ============================================================
    # 3. Check if "Unrelated" pairs might actually be same family
    # ============================================================
    print("\n[3] Checking Unrelated pairs that might be same family...")
    
    # Unrelated but same family number
    unrel_same_fam = unrelated[unrelated['Family1'] == unrelated['Family2']]
    if len(unrel_same_fam) > 0:
        print(f"\n  WARNING: {len(unrel_same_fam)} 'Unrelated' pairs are from SAME FAMILY!")
        output_same = OUTPUT_DIR / "diagnostic_unrelated_but_same_family.csv"
        unrel_same_fam[available_cols].sort_values('IBS', ascending=False).to_csv(output_same, index=False)
        print(f"  Saved: {output_same}")
        
        print("\n  These pairs (same family but labeled Unrelated):")
        print("-" * 100)
        for _, row in unrel_same_fam.sort_values('IBS', ascending=False).iterrows():
            print(f"  {row['Sample1']:<18} {row['Sample2']:<18} Family={row['Family1']} "
                  f"Members: {row['Member1']}-{row['Member2']} IBS={row['IBS']:.4f}")
    else:
        print("  OK: No Unrelated pairs from same family")
    
    # ============================================================
    # 4. Unrelated pairs with unusually high metrics
    # ============================================================
    print("\n[4] Unrelated pairs with metrics suggesting relatedness...")
    
    # Thresholds (rough estimates for relatedness)
    # 3rd degree: ~0.125 kinship, ~0.7 IBS
    # 4th degree: ~0.0625 kinship, ~0.67 IBS
    
    suspicious_unrel = unrelated[
        (unrelated['IBS'] > 0.66) | 
        (unrelated['IBD'] > 0.10) | 
        (unrelated['Kinship'] > 0.05)
    ].copy()
    
    if len(suspicious_unrel) > 0:
        print(f"\n  Found {len(suspicious_unrel)} suspicious Unrelated pairs:")
        output_susp = OUTPUT_DIR / "diagnostic_suspicious_unrelated.csv"
        suspicious_unrel[available_cols].sort_values('IBS', ascending=False).to_csv(output_susp, index=False)
        print(f"  Saved: {output_susp}")
        
        print("\n  Pairs with IBS > 0.66 or IBD > 0.10 or Kinship > 0.05:")
        print("-" * 110)
        print(f"  {'Sample1':<18} {'Sample2':<18} {'Fam1':>5} {'Fam2':>5} {'IBS':>8} {'IBD':>8} {'Kinship':>8} {'Relation':<15}")
        print("-" * 110)
        
        for _, row in suspicious_unrel.sort_values('IBS', ascending=False).head(30).iterrows():
            print(f"  {row['Sample1']:<18} {row['Sample2']:<18} {row['Family1']:>5} {row['Family2']:>5} "
                  f"{row['IBS']:>8.4f} {row['IBD']:>8.4f} {row['Kinship']:>8.4f} {row['Relationship']:<15}")
    else:
        print("  No suspicious Unrelated pairs found")
    
    # ============================================================
    # 5. Summary statistics by Relationship
    # ============================================================
    print("\n[5] Summary statistics by Relationship...")
    
    summary = df.groupby('Relationship').agg({
        'IBS': ['count', 'mean', 'std', 'min', 'max'],
        'IBD': ['mean', 'std', 'min', 'max'],
        'Kinship': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    summary = summary.reset_index()
    
    output_summary = OUTPUT_DIR / "diagnostic_relationship_summary.csv"
    summary.to_csv(output_summary, index=False)
    print(f"  Saved: {output_summary}")
    
    print("\n  Relationship Statistics:")
    print("-" * 90)
    print(f"  {'Relationship':<20} {'N':>6} {'IBS_mean':>10} {'IBS_std':>10} {'IBD_mean':>10} {'Kin_mean':>10}")
    print("-" * 90)
    
    for _, row in summary.iterrows():
        print(f"  {row['Relationship']:<20} {int(row['IBS_count']):>6} "
              f"{row['IBS_mean']:>10.4f} {row['IBS_std']:>10.4f} "
              f"{row['IBD_mean']:>10.4f} {row['Kinship_mean']:>10.4f}")
    
    # ============================================================
    # 6. Analyze 2nd degree split (siblings vs grandparent-grandchild)
    # ============================================================
    print("\n[6] Analyzing 2nd degree pairs (Sibling vs Grandparent-Grandchild)...")
    
    second_degree = df[df['Degree'] == 2].copy()
    
    if len(second_degree) > 0:
        output_2nd = OUTPUT_DIR / "diagnostic_2nd_degree_detailed.csv"
        second_degree[available_cols].sort_values('IBS', ascending=False).to_csv(output_2nd, index=False)
        print(f"  Saved: {output_2nd}")
        
        # Group by relationship type within 2nd degree
        rel_types = second_degree.groupby('Relationship').agg({
            'IBS': ['count', 'mean', 'std'],
            'IBD': ['mean', 'std'],
            'Kinship': ['mean', 'std']
        }).round(4)
        
        print("\n  2nd Degree breakdown:")
        for rel in second_degree['Relationship'].unique():
            rel_data = second_degree[second_degree['Relationship'] == rel]
            print(f"    {rel}: n={len(rel_data)}, IBS={rel_data['IBS'].mean():.4f}±{rel_data['IBS'].std():.4f}")
    
    # ============================================================
    # 7. Analyze 3rd degree pairs
    # ============================================================
    print("\n[7] Analyzing 3rd degree pairs...")
    
    third_degree = df[df['Degree'] == 3].copy()
    
    if len(third_degree) > 0:
        output_3rd = OUTPUT_DIR / "diagnostic_3rd_degree_detailed.csv"
        third_degree[available_cols].sort_values('IBS', ascending=False).to_csv(output_3rd, index=False)
        print(f"  Saved: {output_3rd}")
        
        print(f"\n  3rd Degree: n={len(third_degree)}, "
              f"IBS={third_degree['IBS'].mean():.4f}±{third_degree['IBS'].std():.4f}")
    
    # ============================================================
    # 8. Export each degree separately sorted by IBS
    # ============================================================
    print("\n[8] Exporting each degree separately...")
    
    for degree in sorted(df['Degree'].unique()):
        degree_df = df[df['Degree'] == degree].copy()
        degree_df = degree_df.sort_values('IBS', ascending=False)
        
        degree_label = {
            0: "unrelated", 1: "1st", 2: "2nd", 3: "3rd", 
            4: "4th", 5: "5th", 6: "6th"
        }.get(degree, str(degree))
        
        output_deg = OUTPUT_DIR / f"diagnostic_degree_{degree_label}_sorted_by_IBS.csv"
        degree_df[available_cols].to_csv(output_deg, index=False)
        print(f"  Degree {degree} ({degree_label}): {len(degree_df)} pairs -> {output_deg.name}")
    
    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("DIAGNOSTIC FILES CREATED")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("\nKey files to check:")
    print("  1. diagnostic_suspicious_unrelated.csv - Unrelated with high metrics")
    print("  2. diagnostic_unrelated_but_same_family.csv - Same family but labeled Unrelated")
    print("  3. diagnostic_unrelated_sorted_by_IBS.csv - All Unrelated sorted by IBS")
    print("  4. diagnostic_2nd_degree_detailed.csv - 2nd degree details")
    print("  5. diagnostic_relationship_summary.csv - Summary stats")


if __name__ == "__main__":
    main()