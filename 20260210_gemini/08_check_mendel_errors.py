#!/usr/bin/env python3
"""
08_check_mendel_errors.py
- PLINK --mendel 옵션을 사용하여 가족 내 유전 오류 검사 스크립트 생성
- 문제의 FAM004, FAM009가 실제로 Error Rate가 높은지 확인 가능
"""

from pathlib import Path
import os

# 설정
HOME = Path.home()
WORK_DIR = HOME / "kinship/Analysis/20251031_wgrs/06_kinship_analysis"
PLINK_DIR = WORK_DIR / "plink_files"  # PLINK 바이너리 포맷 파일들이 있는 곳
OUT_DIR = WORK_DIR / "qc_mendel"
SCRIPTS_DIR = WORK_DIR / "scripts"

OUT_DIR.mkdir(parents=True, exist_ok=True)

def create_mendel_script():
    script_path = SCRIPTS_DIR / "run_mendel_check.sh"
    
    with open(script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Check Mendelian Errors per Marker Set\n\n")
        
        # 분석할 마커셋들
        marker_sets = ["NFS_36K", "NFS_24K", "NFS_20K", "NFS_12K", "NFS_6K"]
        
        for marker in marker_sets:
            input_base = PLINK_DIR / f"{marker}"
            output_base = OUT_DIR / f"{marker}_mendel"
            
            f.write(f"echo 'Checking {marker} ...'\n")
            # --mendel: Mendel error check
            # --allow-no-sex: 성별 정보 없어도 실행
            f.write(f"plink --bfile {input_base} --mendel --allow-no-sex --out {output_base}\n")
            f.write("echo 'Done.'\n\n")
            
    os.chmod(script_path, 0o755)
    print(f"Created script: {script_path}")
    print("\nRun this script to generate .mendel (error per family) and .lmendel (error per marker) files.")
    print(f"bash {script_path}")
    
    print("\n[Analysis Guide]")
    print("1. Check *.fmendel files: See if FAM004/FAM009 have significantly higher errors.")
    print("2. Check *.lmendel files: Identify markers with high error counts.")

if __name__ == "__main__":
    create_mendel_script()