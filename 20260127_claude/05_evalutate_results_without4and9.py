#!/usr/bin/env python3
"""
용역회사 보고서 비교용 - Second-Cousin (6촌) ROC AUC 계산
EXCLUDING families 004 and 009

보고서 정의:
- By Closer Relations: 6촌 vs 1-5촌 (더 가까운 혈연)
- By More Distant Relations: 6촌 vs Unrelated (비혈연)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
HOME = Path.home()
WORK_DIR = HOME / "kinship/Analysis/20251031_wgrs/06_kinship_analysis_without4and9"
RESULTS_DIR = WORK_DIR / "results"
GROUND_TRUTH = WORK_DIR / "family_relationships.csv"

MARKER_SETS = ['NFS_36K', 'NFS_24K', 'NFS_20K', 'NFS_12K', 'NFS_6K', 'Kintelligence', 'QIAseq']

# 용역회사 보고서 수치 (비교용)
REPORT_VALUES = {
    'NFS_36K': {'closer': {'IBD': 0.992, 'IBS': 0.991, 'Kinship': 0.991},
                'distant': {'IBD': 0.930, 'IBS': 0.954, 'Kinship': 0.939}},
    'NFS_24K': {'closer': {'IBD': 0.989, 'IBS': 0.990, 'Kinship': 0.988},
                'distant': {'IBD': 0.914, 'IBS': 0.953, 'Kinship': 0.920}},
    'NFS_12K': {'closer': {'IBD': 0.991, 'IBS': 0.991, 'Kinship': 0.991},
                'distant': {'IBD': 0.841, 'IBS': 0.855, 'Kinship': 0.857}},
    'NFS_6K':  {'closer': {'IBD': 0.984, 'IBS': 0.987, 'Kinship': 0.983},
                'distant': {'IBD': 0.806, 'IBS': 0.810, 'Kinship': 0.765}},
    'Kintelligence': {'closer': {'IBD': 0.987, 'IBS': 0.990, 'Kinship': 0.984},
                      'distant': {'IBD': 0.888, 'IBS': 0.914, 'Kinship': 0.863}},
}


def load_plink_genome(filepath):
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath, delim_whitespace=True)
    df['Sample1'] = df['IID1'].astype(str)
    df['Sample2'] = df['IID2'].astype(str)
    df['pair'] = df.apply(lambda r: tuple(sorted([r['Sample1'], r['Sample2']])), axis=1)
    return df[['pair', 'PI_HAT', 'DST']]


def load_king_kinship(filepath):
    for ext in ['.kin0', '.kin']:
        test_path = filepath.with_suffix(ext)
        if test_path.exists():
            filepath = test_path
            break
    else:
        return None
    df = pd.read_csv(filepath, sep='\t')
    col = 'ID1' if 'ID1' in df.columns else 'IID1'
    df['Sample1'] = df[col].astype(str)
    df['Sample2'] = df[col.replace('1', '2')].astype(str)
    df['pair'] = df.apply(lambda r: tuple(sorted([r['Sample1'], r['Sample2']])), axis=1)
    return df[['pair', 'Kinship']]


def merge_results(ground_truth, marker_set):
    plink_file = RESULTS_DIR / f"{marker_set}_plink.genome"
    king_file = RESULTS_DIR / f"{marker_set}_king.kin0"
    
    plink_df = load_plink_genome(plink_file)
    king_df = load_king_kinship(king_file)
    
    gt = ground_truth.copy()
    gt['pair'] = gt.apply(lambda r: tuple(sorted([str(r['Sample1']), str(r['Sample2'])])), axis=1)
    
    if plink_df is not None:
        plink_df = plink_df.rename(columns={'PI_HAT': 'IBD', 'DST': 'IBS'})
        gt = gt.merge(plink_df[['pair', 'IBD', 'IBS']], on='pair', how='left')
    else:
        gt['IBD'] = np.nan
        gt['IBS'] = np.nan
    
    if king_df is not None:
        gt = gt.merge(king_df[['pair', 'Kinship']], on='pair', how='left')
    else:
        gt['Kinship'] = np.nan
    
    gt['Marker_Set'] = marker_set
    return gt


def calculate_auc(y_true, y_score):
    """Calculate ROC AUC, handling NaN values"""
    valid_mask = ~np.isnan(y_score)
    y_true = np.array(y_true)[valid_mask]
    y_score = np.array(y_score)[valid_mask]
    
    if len(np.unique(y_true)) < 2 or len(y_true) == 0:
        return None
    
    try:
        return roc_auc_score(y_true, y_score)
    except:
        return None


def main():
    print("=" * 90)
    print("용역회사 보고서 비교 - Second-Cousin (6촌) ROC AUC")
    print("(Families 004, 009 제외)")
    print("=" * 90)
    
    # Load ground truth
    print("\n[1] Loading ground truth...")
    if not GROUND_TRUTH.exists():
        print(f"  ERROR: {GROUND_TRUTH} not found!")
        print("  Run 04_ground_truth_without4and9.py first!")
        return
    
    ground_truth = pd.read_csv(GROUND_TRUTH)
    
    # Count relationships
    second_cousin = ground_truth[ground_truth['Degree'] == 6]
    closer_relations = ground_truth[ground_truth['Degree'].isin([1, 2, 3, 4, 5])]
    unrelated = ground_truth[ground_truth['Is_Related'] == False]
    
    print(f"  Second-Cousin (6촌): {len(second_cousin)} pairs")
    print(f"  Closer Relations (1-5촌): {len(closer_relations)} pairs")
    print(f"  Unrelated: {len(unrelated)} pairs")
    
    # Load and merge results
    print("\n[2] Loading results for each marker set...")
    all_results = []
    
    for marker_set in MARKER_SETS:
        print(f"  {marker_set}...", end=" ")
        merged = merge_results(ground_truth, marker_set)
        all_results.append(merged)
        n_valid = merged['IBS'].notna().sum()
        print(f"{n_valid:,} pairs with data")
    
    all_df = pd.concat(all_results, ignore_index=True)
    
    # Calculate AUC for each scenario
    print("\n[3] Calculating ROC AUC...")
    
    results = {}
    
    for marker_set in MARKER_SETS:
        df = all_df[all_df['Marker_Set'] == marker_set]
        
        # Get Second-Cousin (6촌) data
        second_cousin_data = df[df['Degree'] == 6]
        closer_data = df[df['Degree'].isin([1, 2, 3, 4, 5])]
        unrelated_data = df[df['Is_Related'] == False]
        
        results[marker_set] = {'closer': {}, 'distant': {}}
        
        # By Closer Relations: 6촌 (positive) vs 1-5촌 (negative)
        # 6촌이 더 낮은 값을 가지므로, 1-5촌을 positive로 설정
        combined_closer = pd.concat([closer_data, second_cousin_data])
        y_true_closer = (combined_closer['Degree'].isin([1, 2, 3, 4, 5])).astype(int)
        
        for metric in ['IBD', 'IBS', 'Kinship']:
            auc = calculate_auc(y_true_closer, combined_closer[metric].values)
            results[marker_set]['closer'][metric] = auc
        
        # By More Distant Relations: 6촌 (positive) vs Unrelated (negative)
        # 6촌이 더 높은 값을 가지므로, 6촌을 positive로 설정
        combined_distant = pd.concat([second_cousin_data, unrelated_data])
        y_true_distant = (combined_distant['Degree'] == 6).astype(int)
        
        for metric in ['IBD', 'IBS', 'Kinship']:
            auc = calculate_auc(y_true_distant, combined_distant[metric].values)
            results[marker_set]['distant'][metric] = auc
    
    # Print comparison table
    print("\n" + "=" * 90)
    print("ROC AUC of Second-Cousins (6촌)")
    print("=" * 90)
    
    print("\n" + "-" * 90)
    print(f"{'Marker Set':<15} | {'By Closer Relations (1-5촌 vs 6촌)':<35} | {'By More Distant Relations (6촌 vs Unrel)':<35}")
    print(f"{'':15} | {'IBD':>10} {'IBS':>10} {'Kinship':>10} | {'IBD':>10} {'IBS':>10} {'Kinship':>10}")
    print("-" * 90)
    
    for marker_set in MARKER_SETS:
        r = results[marker_set]
        
        # Format values
        c_ibd = f"{r['closer']['IBD']:.3f}" if r['closer']['IBD'] else "N/A"
        c_ibs = f"{r['closer']['IBS']:.3f}" if r['closer']['IBS'] else "N/A"
        c_kin = f"{r['closer']['Kinship']:.3f}" if r['closer']['Kinship'] else "N/A"
        
        d_ibd = f"{r['distant']['IBD']:.3f}" if r['distant']['IBD'] else "N/A"
        d_ibs = f"{r['distant']['IBS']:.3f}" if r['distant']['IBS'] else "N/A"
        d_kin = f"{r['distant']['Kinship']:.3f}" if r['distant']['Kinship'] else "N/A"
        
        print(f"{marker_set:<15} | {c_ibd:>10} {c_ibs:>10} {c_kin:>10} | {d_ibd:>10} {d_ibs:>10} {d_kin:>10}")
    
    # Compare with report
    print("\n" + "=" * 90)
    print("비교: 내 결과 vs 용역회사 보고서")
    print("=" * 90)
    
    print("\n[By Closer Relations]")
    print(f"{'Marker Set':<15} | {'IBD':^21} | {'IBS':^21} | {'Kinship':^21}")
    print(f"{'':15} | {'Mine':>8} {'Report':>8} {'Diff':>5} | {'Mine':>8} {'Report':>8} {'Diff':>5} | {'Mine':>8} {'Report':>8} {'Diff':>5}")
    print("-" * 90)
    
    for marker_set in ['NFS_36K', 'NFS_24K', 'NFS_12K', 'NFS_6K', 'Kintelligence']:
        r = results[marker_set]
        rep = REPORT_VALUES[marker_set]['closer']
        
        row = f"{marker_set:<15} |"
        for metric in ['IBD', 'IBS', 'Kinship']:
            mine = r['closer'][metric]
            report = rep[metric]
            if mine:
                diff = mine - report
                diff_str = f"{diff:+.3f}" if abs(diff) < 0.1 else f"{diff:+.3f}*"
                row += f" {mine:>8.3f} {report:>8.3f} {diff_str:>5} |"
            else:
                row += f" {'N/A':>8} {report:>8.3f} {'N/A':>5} |"
        print(row)
    
    print("\n[By More Distant Relations]")
    print(f"{'Marker Set':<15} | {'IBD':^21} | {'IBS':^21} | {'Kinship':^21}")
    print(f"{'':15} | {'Mine':>8} {'Report':>8} {'Diff':>5} | {'Mine':>8} {'Report':>8} {'Diff':>5} | {'Mine':>8} {'Report':>8} {'Diff':>5}")
    print("-" * 90)
    
    for marker_set in ['NFS_36K', 'NFS_24K', 'NFS_12K', 'NFS_6K', 'Kintelligence']:
        r = results[marker_set]
        rep = REPORT_VALUES[marker_set]['distant']
        
        row = f"{marker_set:<15} |"
        for metric in ['IBD', 'IBS', 'Kinship']:
            mine = r['distant'][metric]
            report = rep[metric]
            if mine:
                diff = mine - report
                diff_str = f"{diff:+.3f}" if abs(diff) < 0.1 else f"{diff:+.3f}*"
                row += f" {mine:>8.3f} {report:>8.3f} {diff_str:>5} |"
            else:
                row += f" {'N/A':>8} {report:>8.3f} {'N/A':>5} |"
        print(row)
    
    # Sample counts
    print("\n" + "=" * 90)
    print("Sample Counts")
    print("=" * 90)
    print(f"  Second-Cousin (6촌): {len(second_cousin)} pairs")
    print(f"  Closer Relations (1-5촌): {len(closer_relations)} pairs")
    print(f"  Unrelated: {len(unrelated)} pairs")
    print(f"  보고서: Closer=160, Distant=2,992")
    
    # Save results
    results_df = []
    for marker_set in MARKER_SETS:
        r = results[marker_set]
        for scenario in ['closer', 'distant']:
            for metric in ['IBD', 'IBS', 'Kinship']:
                results_df.append({
                    'Marker_Set': marker_set,
                    'Scenario': scenario,
                    'Metric': metric,
                    'AUC': r[scenario][metric]
                })
    
    results_df = pd.DataFrame(results_df)
    output_file = WORK_DIR / "second_cousin_roc_comparison.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n  Results saved: {output_file}")


if __name__ == "__main__":
    main()