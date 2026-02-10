#!/usr/bin/env python3
"""
05_evaluate_results_detailed.py
- 기존 평가 코드를 보완하여 '촌수(Degree)별' 정확도 분석
- 1촌 vs 2촌, 2촌 vs 3촌 등 경계선상 구분 능력 평가
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, accuracy_score

# 설정
HOME = Path.home()
WORK_DIR = HOME / "kinship/Analysis/20251031_wgrs/06_kinship_analysis"
RESULTS_DIR = WORK_DIR / "results"
GT_FILE = WORK_DIR / "family_relationships.csv"

MARKER_SETS = ["NFS_36K", "NFS_24K", "NFS_20K", "NFS_12K", "NFS_6K", "Kintelligence", "QIAseq"]

def load_data():
    gt = pd.read_csv(GT_FILE)
    # Key 생성 (Sample1-Sample2)
    gt['pair_key'] = gt.apply(lambda x: tuple(sorted([x['Sample1'], x['Sample2']])), axis=1)
    return gt.set_index('pair_key')

def analyze_degree_separation(gt_df):
    print("="*80)
    print("DETAILED DEGREE SEPARATION ANALYSIS")
    print("="*80)
    
    for marker in MARKER_SETS:
        result_file = RESULTS_DIR / f"kinship_results_{marker}.csv"
        if not result_file.exists():
            continue
            
        res = pd.read_csv(result_file)
        res['pair_key'] = res.apply(lambda x: tuple(sorted([x['ID1'], x['ID2']])), axis=1)
        res = res.set_index('pair_key')
        
        # Merge
        merged = gt_df.join(res, how='inner', lsuffix='_gt', rsuffix='_res')
        
        # Kinship 계수를 이용한 추정 촌수 계산 (Cutoff 기준)
        # 일반적인 Cutoff:
        # 1촌: [0.177, 0.354], 2촌: [0.0884, 0.177], 3촌: [0.0442, 0.0884]
        # 이는 예시이며, 실제로는 분포를 보고 최적화해야 함
        
        def infer_degree(k):
            if k > 0.354: return 0 # Duplicate/Twin
            if k > 0.177: return 1
            if k > 0.0884: return 2
            if k > 0.0442: return 3
            return 4 # 4촌 이상 or 남남
            
        merged['Predicted_Degree'] = merged['Kinship'].apply(infer_degree)
        
        # 3촌 이내 데이터만 필터링해서 정확도 확인
        target_degrees = [1, 2, 3]
        subset = merged[merged['Degree'].isin(target_degrees)]
        
        if len(subset) == 0:
            continue

        acc = accuracy_score(subset['Degree'], subset['Predicted_Degree'])
        
        print(f"\n[ {marker} ] Accuracy (1st~3rd Degree): {acc:.4f}")
        
        # Confusion Matrix (1, 2, 3촌 간의 오분류 확인)
        cm = confusion_matrix(subset['Degree'], subset['Predicted_Degree'], labels=[1, 2, 3, 4])
        print("  Confusion Matrix (Row: True, Col: Pred) [1, 2, 3, 4+]")
        print(cm)
        
        # 2촌 vs 3촌 구분 능력 (Specific Metric)
        deg23 = merged[merged['Degree'].isin([2, 3])]
        if len(deg23) > 0:
            # 단순 평균 비교
            mean_2 = deg23[deg23['Degree']==2]['Kinship'].mean()
            std_2 = deg23[deg23['Degree']==2]['Kinship'].std()
            mean_3 = deg23[deg23['Degree']==3]['Kinship'].mean()
            std_3 = deg23[deg23['Degree']==3]['Kinship'].std()
            
            # Separation (Distance between means / sum of stds)
            sep = (mean_2 - mean_3) / (std_2 + std_3)
            print(f"  2nd vs 3rd Separation Score: {sep:.4f} (Higher is better)")

if __name__ == "__main__":
    gt = load_data()
    analyze_degree_separation(gt)