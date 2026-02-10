#!/usr/bin/env python3
"""
Step 3: Run Kinship Analysis (PLINK IBS/IBD + KING Kinship)
EXCLUDING families 004 and 009

Uses existing joint VCF and marker BED files, but filters samples
"""

import subprocess
from pathlib import Path

# ============================================================
# Configuration
# ============================================================
HOME = Path.home()
WORK_DIR = HOME / "kinship/Analysis/20251031_wgrs"

# Input
JOINT_VCF = WORK_DIR / "05_jointcall/joint_called.allsites.vcf.gz"
MARKER_BED_DIR = WORK_DIR / "06_kinship_analysis/marker_beds"

# Output - new directory
OUTPUT_DIR = WORK_DIR / "06_kinship_analysis_without4and9"
RESULTS_DIR = OUTPUT_DIR / "results"
VCF_SUBSET_DIR = OUTPUT_DIR / "vcf_subsets"
PLINK_DIR = OUTPUT_DIR / "plink_files"
SCRIPTS_DIR = OUTPUT_DIR / "scripts"

for d in [OUTPUT_DIR, RESULTS_DIR, VCF_SUBSET_DIR, PLINK_DIR, SCRIPTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Samples to EXCLUDE (004 and 009 families)
EXCLUDE_PATTERNS = ['2024-004-', '2024-009-']

# Marker sets
MARKER_SETS = ['NFS_36K', 'NFS_24K', 'NFS_20K', 'NFS_12K', 'NFS_6K', 'Kintelligence', 'QIAseq']

# Tools
BCFTOOLS = "bcftools"
PLINK = "plink"
KING = "king"


def get_samples_to_keep():
    """Get list of samples from VCF, excluding 004 and 009 families"""
    
    print("  Getting sample list from VCF...")
    result = subprocess.run(
        f"{BCFTOOLS} query -l {JOINT_VCF}",
        shell=True, capture_output=True, text=True
    )
    
    all_samples = result.stdout.strip().split('\n')
    print(f"  Total samples in VCF: {len(all_samples)}")
    
    # Filter out 004 and 009 families
    keep_samples = []
    excluded_samples = []
    
    for sample in all_samples:
        exclude = False
        for pattern in EXCLUDE_PATTERNS:
            if pattern in sample:
                exclude = True
                break
        
        if exclude:
            excluded_samples.append(sample)
        else:
            keep_samples.append(sample)
    
    print(f"  Excluding {len(excluded_samples)} samples from families 004/009:")
    for s in excluded_samples:
        print(f"    - {s}")
    print(f"  Keeping {len(keep_samples)} samples")
    
    return keep_samples


def create_sample_file(samples):
    """Create file with samples to keep"""
    sample_file = OUTPUT_DIR / "samples_to_keep.txt"
    with open(sample_file, 'w') as f:
        for s in samples:
            f.write(f"{s}\n")
    return sample_file


def create_analysis_script(marker_set, bed_file, sample_file):
    """Create analysis script for one marker set"""
    
    script_content = f"""#!/bin/bash
#$ -S /bin/bash
#$ -N kinship_{marker_set}_no4and9
#$ -cwd
#$ -pe smp 4
#$ -l h_vmem=16G

echo "======================================"
echo "Kinship Analysis: {marker_set} (without 004/009)"
echo "======================================"
echo "Start: $(date)"

MARKER_SET="{marker_set}"
BED_FILE="{bed_file}"
SAMPLE_FILE="{sample_file}"
VCF_IN="{JOINT_VCF}"
VCF_OUT="{VCF_SUBSET_DIR}/{marker_set}.vcf.gz"
PLINK_OUT="{PLINK_DIR}/{marker_set}"
RESULTS="{RESULTS_DIR}"

# Step 1: Extract marker regions AND filter samples
echo "[1] Extracting markers and filtering samples..."
bcftools view -R $BED_FILE -S $SAMPLE_FILE -Oz -o $VCF_OUT $VCF_IN
bcftools index -t $VCF_OUT

# Check sample count
N_SAMPLES=$(bcftools query -l $VCF_OUT | wc -l)
echo "  Samples in filtered VCF: $N_SAMPLES"

# Step 2: Convert to PLINK format
echo "[2] Converting to PLINK format..."
plink --vcf $VCF_OUT \\
      --double-id \\
      --allow-extra-chr \\
      --make-bed \\
      --out $PLINK_OUT

# Step 3: PLINK IBS/IBD
echo "[3] Calculating PLINK IBS/IBD..."
plink --bfile $PLINK_OUT \\
      --genome \\
      --allow-extra-chr \\
      --out $RESULTS/${{MARKER_SET}}_plink

# Step 4: KING kinship
echo "[4] Calculating KING kinship..."
king -b ${{PLINK_OUT}}.bed \\
     --kinship \\
     --prefix $RESULTS/${{MARKER_SET}}_king

echo "======================================"
echo "End: $(date)"
echo "======================================"
"""
    
    script_path = SCRIPTS_DIR / f"run_{marker_set}.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    return script_path


def main():
    print("=" * 70)
    print("Kinship Analysis Setup (EXCLUDING Families 004 and 009)")
    print("=" * 70)
    
    # Check inputs
    print("\n[1] Checking inputs...")
    
    if not JOINT_VCF.exists():
        print(f"  ERROR: Joint VCF not found: {JOINT_VCF}")
        return
    print(f"  Joint VCF: OK")
    
    if not MARKER_BED_DIR.exists():
        print(f"  ERROR: Marker BED directory not found: {MARKER_BED_DIR}")
        return
    print(f"  Marker BED dir: OK")
    
    # Get samples to keep
    print("\n[2] Getting samples to keep...")
    keep_samples = get_samples_to_keep()
    sample_file = create_sample_file(keep_samples)
    print(f"  Sample file: {sample_file}")
    
    # Create scripts for each marker set
    print("\n[3] Creating analysis scripts...")
    scripts = []
    
    for marker_set in MARKER_SETS:
        bed_file = MARKER_BED_DIR / f"{marker_set}.bed"
        
        if not bed_file.exists():
            print(f"  WARNING: {bed_file} not found, skipping")
            continue
        
        script = create_analysis_script(marker_set, bed_file, sample_file)
        scripts.append(script)
        print(f"  Created: {script.name}")
    
    # Create master submit script
    print("\n[4] Creating master submit script...")
    
    master_script = SCRIPTS_DIR / "submit_all.sh"
    with open(master_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all kinship analysis jobs (without 004/009)\n\n")
        
        for script in scripts:
            f.write(f"qsub {script}\n")
        
        f.write("\necho 'All jobs submitted!'\n")
    
    master_script.chmod(0o755)
    print(f"  Master script: {master_script}")
    
    # Also create a local run script (no qsub)
    local_script = SCRIPTS_DIR / "run_all_local.sh"
    with open(local_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Run all kinship analysis locally (without 004/009)\n\n")
        
        for script in scripts:
            f.write(f"echo '>>> Running {script.name}...'\n")
            f.write(f"bash {script}\n")
            f.write("echo ''\n")
        
        f.write("\necho 'All analyses complete!'\n")
    
    local_script.chmod(0o755)
    print(f"  Local script: {local_script}")
    
    # Summary
    print("\n" + "=" * 70)
    print("SETUP COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Samples kept: {len(keep_samples)}")
    print(f"Scripts created: {len(scripts)}")
    
    print("\nTo run:")
    print(f"  Option 1 (qsub): bash {master_script}")
    print(f"  Option 2 (local): bash {local_script}")


if __name__ == "__main__":
    main()