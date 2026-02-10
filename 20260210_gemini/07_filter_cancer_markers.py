#!/usr/bin/env python3
"""
07_filter_cancer_markers.py
- TSV 파일 정보를 바탕으로 36K, 24K 마커셋에서 Cancer Gene을 제거한 BED 파일 생성
- 이를 통해 20K와의 성능 차이가 Cancer Gene 유무 때문인지 검증
"""

import pandas as pd
from pathlib import Path

# 설정
HOME = Path.home()
KINSHIP_DIR = Path("/BiO/Access/ehojune/kinship")
WORK_DIR = HOME / "kinship/Analysis/20251031_wgrs/06_kinship_analysis"
BED_DIR = WORK_DIR / "marker_beds"
TSV_FILE = KINSHIP_DIR / "kinship_marker_with_20K_with_cancerfiltering.tsv"

def filter_markers():
    print("Loading TSV file...")
    df = pd.read_csv(TSV_FILE, sep='\t')

    targets = ['36K', '24K', '12K', '6K']

    # 공통 필터 조건 정의
    no_cancer = df['CANCER_GENE_TYPE'] == 'No'

    protein_coding_exon = (
        (df['REGION'] == 'exon') &
        (df['GENE_TYPE'] == 'protein_coding') &
        (df['TRANSCRIPT_TYPE'] == 'protein_coding')
    )

    for target in targets:
        print(f"\nProcessing NFS_{target}...")

        # 원래 마커
        original_mask = df[target] == 'O'
        original_count = original_mask.sum()

        # 최종 필터 적용
        filtered_mask = (
            original_mask &
            no_cancer &
            (~protein_coding_exon)
        )

        filtered_df = df[filtered_mask]
        filtered_count = len(filtered_df)

        print(f"  - Original: {original_count:,}")
        print(f"  - Filtered (No Cancer + Non PC-Exon): {filtered_count:,}")
        print(f"  - Removed: {original_count - filtered_count:,}")

        bed_filename = f"NFS_{target}_NoCancer_NoPCExon.bed"
        bed_path = BED_DIR / bed_filename

        with open(bed_path, 'w') as f:
            for _, row in filtered_df.iterrows():
                chrom = row['CHROM']
                pos = int(row['POS'])
                f.write(f"{chrom}\t{pos-1}\t{pos}\t{row['RSID']}\n")

        print(f"  - Saved: {bed_path}")


if __name__ == "__main__":
    filter_markers()