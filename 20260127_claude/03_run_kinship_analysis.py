#!/usr/bin/env python3
"""
Step 3: Extract VCF subsets, convert to PLINK, calculate IBS/IBD/Kinship
- For each marker set (NFS 6K~36K, Kintelligence, QIAseq)
- Sample names: 2024-001-1A format (as in VCF)
"""

import subprocess
from pathlib import Path
import os

# ============================================================
# Configuration
# ============================================================
HOME = Path.home()
WORK_DIR = HOME / "kinship/Analysis/20251031_wgrs/06_kinship_analysis"
JOINT_VCF = HOME / "kinship/Analysis/20251031_wgrs/05_jointcall/joint_called.allsites.vcf.gz"
BED_DIR = WORK_DIR / "marker_beds"
VCF_DIR = WORK_DIR / "vcf_subsets"
PLINK_DIR = WORK_DIR / "plink_files"
RESULTS_DIR = WORK_DIR / "results"
SCRIPTS_DIR = WORK_DIR / "scripts"
LOGS_DIR = WORK_DIR / "logs"

# Create directories
for d in [VCF_DIR, PLINK_DIR, RESULTS_DIR, SCRIPTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Marker sets
MARKER_SETS = {
    'NFS_36K': BED_DIR / "NFS_36K_NoCancer_NoPCExon.bed",
    'NFS_24K': BED_DIR / "NFS_24K_NoCancer_NoPCExon.bed",
    'NFS_20K': BED_DIR / "NFS_20K.bed",
    'NFS_12K': BED_DIR / "NFS_12K_NoCancer_NoPCExon.bed",
    'NFS_6K': BED_DIR / "NFS_6K_NoCancer_NoPCExon.bed",
    'Kintelligence': BED_DIR / "Kintelligence.bed",
    'QIAseq': BED_DIR / "QIAseq.bed",
}


def create_analysis_script(name, bed_path):
    """Create qsub script for one marker set"""
    
    vcf_out = VCF_DIR / f"{name}.vcf.gz"
    plink_prefix = PLINK_DIR / name
    results_prefix = RESULTS_DIR / name
    
    script_content = f"""#!/bin/bash
#$ -N kin_{name}
#$ -o {LOGS_DIR}/kin_{name}.out
#$ -e {LOGS_DIR}/kin_{name}.err
#$ -cwd
#$ -V
#$ -pe smp 4

echo "========================================"
echo "Processing: {name}"
echo "========================================"
echo "Start time: $(date)"

# Step 1: Extract VCF subset
echo ""
echo "[1/4] Extracting VCF subset..."
bcftools view -R {bed_path} {JOINT_VCF} -Oz -o {vcf_out}
tabix -p vcf {vcf_out}

VAR_COUNT=$(bcftools view -H {vcf_out} | wc -l)
echo "      Total sites: $VAR_COUNT"

SNP_COUNT=$(bcftools view -H -v snps {vcf_out} | wc -l)
echo "      SNP sites: $SNP_COUNT"

# Step 2: Convert to PLINK
echo ""
echo "[2/4] Converting to PLINK format..."
plink --vcf {vcf_out} \\
      --make-bed \\
      --out {plink_prefix} \\
      --allow-extra-chr \\
      --double-id \\
      --set-missing-var-ids @:#:\\$1:\\$2 \\
      --vcf-half-call m \\
      2>/dev/null

echo "      Variants in PLINK: $(wc -l < {plink_prefix}.bim)"
echo "      Samples in PLINK: $(wc -l < {plink_prefix}.fam)"

# Step 3: Calculate IBS/IBD (PLINK --genome)
echo ""
echo "[3/4] Calculating IBS/IBD (PLINK --genome)..."
plink --bfile {plink_prefix} \\
      --genome \\
      --out {results_prefix}_plink \\
      --allow-extra-chr \\
      2>/dev/null

echo "      Output: {results_prefix}_plink.genome"
PAIRS=$(tail -n +2 {results_prefix}_plink.genome | wc -l)
echo "      Pairs calculated: $PAIRS"

# Step 4: Calculate Kinship Coefficient (KING)
echo ""
echo "[4/4] Calculating Kinship Coefficient (KING)..."
king -b {plink_prefix}.bed \\
     --kinship \\
     --prefix {results_prefix}_king \\
     2>/dev/null

if [ -f {results_prefix}_king.kin0 ]; then
    echo "      Output: {results_prefix}_king.kin0"
    KING_PAIRS=$(tail -n +2 {results_prefix}_king.kin0 | wc -l)
    echo "      Pairs calculated: $KING_PAIRS"
else
    echo "      KING output not found (may need different version)"
fi

echo ""
echo "========================================"
echo "Completed: {name}"
echo "End time: $(date)"
echo "========================================"
"""
    
    script_path = SCRIPTS_DIR / f"analyze_{name}.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    
    return script_path


def main():
    print("=" * 70)
    print("Step 3: Kinship Analysis Pipeline Setup")
    print("=" * 70)
    
    # Check prerequisites
    print("\n[1] Checking prerequisites...")
    
    if not JOINT_VCF.exists():
        print(f"  ERROR: Joint VCF not found: {JOINT_VCF}")
        print("  Please complete joint calling first!")
        return
    print(f"  Joint VCF: {JOINT_VCF} ?")
    
    # Check marker BED files
    print("\n[2] Checking marker BED files...")
    
    valid_sets = {}
    for name, bed_path in MARKER_SETS.items():
        if bed_path.exists():
            count = sum(1 for _ in open(bed_path))
            print(f"  {name:15}: {count:>10,} markers ?")
            valid_sets[name] = bed_path
        else:
            print(f"  {name:15}: NOT FOUND ?")
    
    if not valid_sets:
        print("\n  ERROR: No marker BED files found!")
        print(f"  Please run 00_generate_marker_beds.py first!")
        return
    
    # Create analysis scripts
    print("\n[3] Creating analysis scripts...")
    
    scripts = []
    for name, bed_path in valid_sets.items():
        script = create_analysis_script(name, bed_path)
        scripts.append((name, script))
        print(f"  Created: {script.name}")
    
    # Create master submit script
    master_script = SCRIPTS_DIR / "submit_all_kinship.sh"
    with open(master_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all kinship analysis jobs\n\n")
        for name, script in scripts:
            f.write(f"qsub {script}\n")
            f.write("sleep 1\n")
        f.write("\necho 'All kinship analysis jobs submitted!'\n")
        f.write("echo 'Check status with: qstat'\n")
    os.chmod(master_script, 0o755)
    
    # Create local run script (without qsub)
    local_script = SCRIPTS_DIR / "run_all_kinship_local.sh"
    with open(local_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Run all kinship analyses locally (sequential)\n\n")
        for name, script in scripts:
            f.write(f"echo '\\n>>> Running {name}...'\n")
            f.write(f"bash {script}\n")
        f.write("\necho '\\nAll analyses complete!'\n")
    os.chmod(local_script, 0o755)
    
    # Summary
    print("\n" + "=" * 70)
    print("Setup Complete!")
    print("=" * 70)
    print(f"\n  Scripts directory: {SCRIPTS_DIR}")
    print(f"  Results will be in: {RESULTS_DIR}")
    print(f"\n  Marker sets to analyze: {len(valid_sets)}")
    
    print(f"\n  To submit all jobs (qsub):")
    print(f"    bash {master_script}")
    
    print(f"\n  To run locally (sequential):")
    print(f"    bash {local_script}")
    
    print(f"\n  Or run individually:")
    print(f"    qsub {scripts[0][1]}")


if __name__ == "__main__":
    main()