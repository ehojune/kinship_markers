#!/usr/bin/env python3
"""
Step 1: HaplotypeCaller GVCF calling for all samples
- Marker regions only (merged_markers.bed)
- Output all sites (variant + non-variant) using -ERC BP_RESOLUTION
- Submit via qsub with -V option
"""

import subprocess
from pathlib import Path
import os
import re

# ============================================================
# Configuration
# ============================================================
HOME = Path.home()
WORK_DIR = HOME / "kinship/Analysis/20251031_wgrs"
GVCF_DIR = WORK_DIR / "04_gvcf"
SCRIPTS_DIR = WORK_DIR / "scripts"
LOGS_DIR = WORK_DIR / "logs"

# Reference and intervals
REF = HOME / "kinship/Analysis/references/hg38.fa"
MARKERS_BED = Path("/BiO/Access/ehojune/kinship/merged_markers.bed")

# BAM locations
BAM_DIR_1 = HOME / "kinship/Analysis"  # 2024-004-*, 2024-009-* 
BAM_DIR_2 = HOME / "kinship/Analysis/Aginglab_bam/ReadMapping.to.hg38.by.BWA.mem"  # Others

# GATK path
GATK = "/BiO/Access/ehojune/anaconda3/envs/gatk462/bin/gatk"

# Create directories
for d in [GVCF_DIR, SCRIPTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def find_all_bams():
    """Find all BAM files and extract sample names"""
    samples = {}
    
    # Pattern 1: 2024-004-10f-10.bam -> 2024-004-10f
    # Pattern 2: 2024-001-10f/2024-001-10f.bam -> 2024-001-10f
    
    # Search BAM_DIR_1 (direct BAM files)
    for bam in BAM_DIR_1.glob("2024-*.bam"):
        if bam.name.endswith(".bam") and not bam.name.endswith(".bam.bai"):
            # Extract sample name: 2024-004-10f-10.bam -> 2024-004-10f
            name = bam.stem  # 2024-004-10f-10
            parts = name.split('-')
            if len(parts) >= 4:
                # e.g., 2024-004-10f-10 -> 2024-004-10f
                sample_name = '-'.join(parts[:3])
            else:
                sample_name = name
            samples[sample_name] = bam
    
    # Search BAM_DIR_2 (subdirectory structure)
    for subdir in BAM_DIR_2.iterdir():
        if subdir.is_dir() and subdir.name.startswith("2024-"):
            bam = subdir / f"{subdir.name}.bam"
            if bam.exists():
                sample_name = subdir.name  # Already in correct format
                samples[sample_name] = bam
    
    return samples


def get_sort_key(sample_name):
    """
    Sort key for sample names
    Format: 2024-001-10f -> (1, '10f')
    Family number numeric sort, then member ID
    """
    parts = sample_name.split('-')
    if len(parts) >= 3:
        family_num = int(parts[1])  # '001' -> 1
        member = parts[2]  # '10f'
        # Member sorting: numeric part first, then letter
        match = re.match(r'(\d+)([a-zA-Z]*)', member)
        if match:
            num = int(match.group(1))
            letter = match.group(2).lower()
            return (family_num, num, letter)
    return (999, 999, sample_name)


def create_gvcf_script(sample_name, bam_path, output_gvcf):
    """Create qsub script for HaplotypeCaller"""
    
    script_content = f"""#!/bin/bash
#$ -N gvcf_{sample_name}
#$ -o {LOGS_DIR}/gvcf_{sample_name}.out
#$ -e {LOGS_DIR}/gvcf_{sample_name}.err
#$ -cwd
#$ -V
#$ -pe smp 4

echo "Starting GVCF calling for {sample_name}"
echo "Start time: $(date)"
echo "BAM: {bam_path}"
echo "Output: {output_gvcf}"

# Activate conda environment if needed
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate gatk462

{GATK} HaplotypeCaller \\
    -R {REF} \\
    -I {bam_path} \\
    -L {MARKERS_BED} \\
    -O {output_gvcf} \\
    -ERC BP_RESOLUTION \\
    --native-pair-hmm-threads 4 \\
    --tmp-dir {WORK_DIR}/tmp

# Index the GVCF
{GATK} IndexFeatureFile -I {output_gvcf}

echo "Completed: {sample_name}"
echo "End time: $(date)"
"""
    
    script_path = SCRIPTS_DIR / f"gvcf_{sample_name}.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    return script_path


def main():
    print("=" * 70)
    print("Step 1: HaplotypeCaller GVCF Calling Pipeline")
    print("=" * 70)
    
    # Check prerequisites
    print("\n[1] Checking prerequisites...")
    
    if not REF.exists():
        print(f"  ERROR: Reference not found: {REF}")
        return
    print(f"  Reference: {REF}")
    
    if not MARKERS_BED.exists():
        print(f"  ERROR: Markers BED not found: {MARKERS_BED}")
        print("  Please run the marker merging step first!")
        return
    
    # Count markers
    marker_count = sum(1 for _ in open(MARKERS_BED))
    print(f"  Markers BED: {MARKERS_BED} ({marker_count:,} regions)")
    
    # Find all BAM files
    print("\n[2] Finding BAM files...")
    samples = find_all_bams()
    
    if not samples:
        print("  ERROR: No BAM files found!")
        return
    
    # Sort samples
    sorted_samples = sorted(samples.items(), key=lambda x: get_sort_key(x[0]))
    
    print(f"  Found {len(sorted_samples)} samples:")
    for sample_name, bam_path in sorted_samples:
        print(f"    {sample_name}: {bam_path}")
    
    # Create tmp directory
    tmp_dir = WORK_DIR / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    
    # Generate scripts
    print("\n[3] Generating qsub scripts...")
    scripts = []
    
    for sample_name, bam_path in sorted_samples:
        output_gvcf = GVCF_DIR / f"{sample_name}.g.vcf.gz"
        script = create_gvcf_script(sample_name, bam_path, output_gvcf)
        scripts.append((sample_name, script, output_gvcf))
        print(f"    Created: {script.name}")
    
    # Create master submit script
    master_script = SCRIPTS_DIR / "submit_all_gvcf.sh"
    with open(master_script, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all GVCF calling jobs\n\n")
        for sample_name, script, _ in scripts:
            f.write(f"qsub {script}\n")
            f.write("sleep 1\n")  # Small delay between submissions
        f.write("\necho 'All GVCF jobs submitted!'\n")
        f.write("echo 'Check status with: qstat'\n")
    
    os.chmod(master_script, 0o755)
    
    # Create sample list for joint calling
    sample_list = WORK_DIR / "sample_list.txt"
    with open(sample_list, 'w') as f:
        for sample_name, _, output_gvcf in scripts:
            f.write(f"{sample_name}\t{output_gvcf}\n")
    
    # Summary
    print("\n" + "=" * 70)
    print("Pipeline Setup Complete!")
    print("=" * 70)
    print(f"\n  Output directory: {GVCF_DIR}")
    print(f"  Scripts directory: {SCRIPTS_DIR}")
    print(f"  Logs directory: {LOGS_DIR}")
    print(f"  Sample list: {sample_list}")
    print(f"\n  Total samples: {len(scripts)}")
    print(f"\n  To submit all jobs:")
    print(f"    bash {master_script}")
    print(f"\n  Or submit individually:")
    print(f"    qsub {scripts[0][1]}")


if __name__ == "__main__":
    main()