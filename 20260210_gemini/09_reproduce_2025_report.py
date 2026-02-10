#!/usr/bin/env python3
"""
09_reproduce_2025_report.py
- FAM004, FAM009를 결과에서 완벽히 제외하고 딱 80명(3,160쌍)으로 2025년 보고서 재현
- Close(160), 2nd Cousin(8), Distant(2,992) 카테고리 매칭
"""

import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

WORK_DIR = Path.home() / "kinship/Analysis/20251031_wgrs/06_kinship_analysis"
RESULTS_DIR = WORK_DIR / "results"
GT_FILE = WORK_DIR / "family_relationships.csv"
MARKER_SETS = ["NFS_36K", "NFS_24K", "NFS_20K", "NFS_12K", "NFS_6K", "Kintelligence", "QIAseq"]

def main():
    print("=" * 80)
    print("2025 REPORT REPRODUCTION (EXCLUDING FAM004 & FAM009)")
    print("=" * 80)

    # 1. Load Ground Truth
    gt = pd.read_csv(GT_FILE)
    # 완전한 제외 처리
    gt = gt[~gt['Family1'].isin(['004', '009']) & ~gt['Family2'].isin(['004', '009'])]
    
    # 조인용 Key 생성
    gt['join_id1'] = gt['Family1'].astype(str) + "_" + gt['Member1']
    gt['join_id2'] = gt['Family2'].astype(str) + "_" + gt['Member2']
    
    gt['pair_key'] = gt.apply(
        lambda x: tuple(sorted([x['join_id1'], x['join_id2']])),
        axis=1
    )
    gt = gt.set_index('pair_key')


    for marker in MARKER_SETS:
        res_file = RESULTS_DIR / f"{marker}_king.kin0"
        if not res_file.exists():
            continue

        res = pd.read_csv(res_file, sep='\t')
        
        # 컬럼명 유연성 확보 (ID1/ID2 or Sample1/Sample2)
        c1 = 'ID1' if 'ID1' in res.columns else 'Sample1'
        c2 = 'ID2' if 'ID2' in res.columns else 'Sample2'
        
        # FAM004, 009 제외 (결과 파일에서도)
        res = res[~res[c1].astype(str).str.contains('004|009') & ~res[c2].astype(str).str.contains('004|009')]
        
        res['join_id1'] = res['FID1'].astype(str) + "_" + res['ID1']
        res['join_id2'] = res['FID2'].astype(str) + "_" + res['ID2']
        
        res['pair_key'] = res.apply(
            lambda x: tuple(sorted([x['join_id1'], x['join_id2']])),
            axis=1
        )
        res = res.set_index('pair_key')

        
        # Inner Join: VCF에 없는 가상 조상 샘플들이 여기서 자동 탈락되어 정확히 80명(3160쌍)만 남음
        merged = gt.join(res, how='inner')
        if len(merged) == 0: continue

        # 2025 보고서 기준 카테고리 분리
        # Second-Cousin (6촌): Degree 6
        merged['Category'] = 'Distant'
        merged.loc[(merged['Degree'] >= 1) & (merged['Degree'] <= 5), 'Category'] = 'Closer'
        merged.loc[merged['Degree'] == 6, 'Category'] = 'Second'


        # Close (1~3촌): Degree 1~3 (통상적으로 8가족 * 20쌍 = 160쌍 나옴)
        merged.loc[(merged['Degree'] >= 1) & (merged['Degree'] <= 3), 'Category'] = 'Close'

        close_n = len(merged[merged['Category']=='Close'])
        sc_n = len(merged[merged['Category']=='Second-Cousin'])
        dist_n = len(merged[merged['Category']=='Distant'])

        print(f"\n[{marker}] Total Pairs: {len(merged)} (Close: {close_n}, 2nd-Cousin: {sc_n}, Distant: {dist_n})")
        
        # 2025 보고서 로직: Close vs Distant / 2nd-Cousin vs Distant
        try:
            # 1. Close vs Distant
            sub_closer = merged[merged['Category'].isin(['Second', 'Closer'])].copy()
            sub_closer['Target'] = (sub_closer['Category'] == 'Second').astype(int)


            
            # 2. Second-Cousin vs Distant
            sub_distant = merged[merged['Category'].isin(['Second', 'Distant'])].copy()
            sub_distant['Target'] = (sub_distant['Category'] == 'Second').astype(int)



      METRIC_MAP = {
          'IBD': ['PI_HAT', 'IBD_PI_HAT'],   # NFS 계열
          'IBS': ['IBS', 'IBS0'],            # IBS0 = 낮을수록 related
          'Kinship': ['Kinship']
      }

            
            for m in metrics:
                auc_closer = roc_auc_score(sub_closer['Target'], sub_closer[m])
                auc_distant = roc_auc_score(sub_distant['Target'], sub_distant[m])
            
                print(
                    f"  - {m:>7} | "
                    f"By Closer: {auc_closer:.3f} | "
                    f"By More Distant: {auc_distant:.3f}"
                )


        except Exception as e:
            print(f"  - Could not calculate AUC: {e}")

if __name__ == "__main__":
    main()