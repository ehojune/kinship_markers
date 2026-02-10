#!/usr/bin/env python3
"""
Step 2: Joint Genotyping - GenomicsDBImport + GenotypeGVCFs
- Combine all GVCFs into GenomicsDB
- Run GenotypeGVCFs with --all-sites for non-variant positions
- Output sorted by family order
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
JOINTCALL_DIR = WORK_DIR / "05_jointcall"
SCRIPTS_DIR = WORK_DIR / "scripts"
LOGS_DIR = WORK_DIR / "logs"

# Reference and intervals
REF = HOME / "kinship/Analysis/references/hg38.fa"
MARKERS_BED = Path("/BiO/Access/ehojune/kinship/merged_markers.bed")

# GATK path
GATK = "/BiO/Access/ehojune/anaconda3/envs/gatk462/bin/gatk"

# Output
GENOMICSDB_DIR = JOINTCALL_DIR / "genomicsdb"
OUTPUT_VCF = JOINTCALL_DIR / "joint_called.allsites.vcf.gz"

# Create directories
for d in [JOINTCALL_DIR, SCRIPTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def get_sort_key(sample_name):
    """
    Sort key for sample names
    Format: 2024-001-10f -> (1, 10, 'f')
    """
    parts = sample_name.split('-')
    if len(parts) >= 3:
        family_num = int(parts[1])
        member = parts[2]
        match = re.match(r'(\d+)([a-zA-Z]*)', member)
        if match:
            num = int(match.group(1))
            letter = match.group(2).lower()
            return (family_num, num, letter)
    return (999, 999, sample_name)


def find_gvcfs():
    """Find all GVCF files and sort by family/member order"""
    gvcfs = {}
    
    for gvcf in GVCF_DIR.glob("*.g.vcf.gz"):
        sample_name = gvcf.stem.replace('.g.vcf', '')
        gvcfs[sample_name] = gvcf
    
    # Sort by family order
    sorted_gvcfs = sorted(gvcfs.items(), key=lambda x: get_sort_key(x[0]))
    return sorted_gvcfs


def create_sample_map(sorted_gvcfs):
    """Create sample map file for GenomicsDBImport"""
    sample_map = JOINTCALL_DIR / "sample_map.txt"
    
    with open(sample_map, 'w') as f:
        for sample_name, gvcf_path in sorted_gvcfs:
            f.write(f"{sample_name}\t{gvcf_path}\n")
    
    return sample_map


def create_interval_list():
    """Convert BED to interval list format for GATK"""
    interval_list = JOINTCALL_DIR / "markers.interval_list"
    
    # Read BED and convert to interval format
    with open(MARKERS_BED, 'r') as bed, open(interval_list, 'w') as out:
        for line in bed:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                chrom, start, end = parts[0], int(parts[1]) + 1, int(parts[2])  # BED is 0-based
                out.write(f"{chrom}:{start}-{end}\n")
    
    return interval_list


def create_jointcall_script(sample_map, interval_list, sorted_gvcfs):
    """Create joint calling script"""
    
    # Generate -V arguments for GenotypeGVCFs (alternative method)
    v_args = ""
    for sample_name, gvcf_path in sorted_gvcfs:
        v_args += f"    -V {gvcf_path} \\\n"
    
    script_content = f"""#!/bin/bash
#$ -N joint_call
#$ -o {LOGS_DIR}/joint_call.out
#$ -e {LOGS_DIR}/joint_call.err
#$ -cwd
#$ -V
#$ -pe smp 8
#$ -l h_vmem=32G

echo "=========================================="
echo "Step 2: Joint Genotyping"
echo "=========================================="
echo "Start time: $(date)"
echo "Sample map: {sample_map}"
echo "Interval list: {interval_list}"

# Create temp directory
TMP_DIR="{WORK_DIR}/tmp_jointcall"
mkdir -p $TMP_DIR

# ============================================================
# Option A: Using GenomicsDBImport (Recommended for many samples)
# ============================================================
echo ""
echo "[1/3] Running GenomicsDBImport..."

# Remove existing genomicsdb if present
rm -rf {GENOMICSDB_DIR}

{GATK} --java-options "-Xmx24g -Xms4g" GenomicsDBImport \\
    --sample-name-map {sample_map} \\
    -L {MARKERS_BED} \\
    --genomicsdb-workspace-path {GENOMICSDB_DIR} \\
    --batch-size 50 \\
    --reader-threads 4 \\
    --tmp-dir $TMP_DIR \\
    --merge-input-intervals

if [ $? -ne 0 ]; then
    echo "ERROR: GenomicsDBImport failed!"
    exit 1
fi

echo ""
echo "[2/3] Running GenotypeGVCFs..."

{GATK} --java-options "-Xmx24g -Xms4g" GenotypeGVCFs \\
    -R {REF} \\
    -V gendb://{GENOMICSDB_DIR} \\
    -O {OUTPUT_VCF} \\
    --include-non-variant-sites \\
    --tmp-dir $TMP_DIR

if [ $? -ne 0 ]; then
    echo "ERROR: GenotypeGVCFs failed!"
    exit 1
fi

echo ""
echo "[3/3] Indexing output VCF..."

{GATK} IndexFeatureFile -I {OUTPUT_VCF}

# ============================================================
# Summary
# ============================================================
echo ""
echo "=========================================="
echo "Joint Calling Complete!"
echo "=========================================="
echo "Output VCF: {OUTPUT_VCF}"
echo ""

# Count variants
TOTAL=$(bcftools view -H {OUTPUT_VCF} | wc -l)
VARIANTS=$(bcftools view -H -v snps,indels {OUTPUT_VCF} | wc -l)
NON_VAR=$((TOTAL - VARIANTS))

echo "Total sites: $TOTAL"
echo "Variant sites: $VARIANTS"
echo "Non-variant sites: $NON_VAR"
echo ""

# Sample count and order
echo "Samples in VCF (in order):"
bcftools query -l {OUTPUT_VCF}

echo ""
echo "End time: $(date)"

# Cleanup
rm -rf $TMP_DIR
"""
    
    script_path = SCRIPTS_DIR / "02_joint_genotyping.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    return script_path


def create_alternative_script(sorted_gvcfs):
    """
    Alternative: CombineGVCFs + GenotypeGVCFs
    Better for smaller sample sets
    """
    
    combined_gvcf = JOINTCALL_DIR / "combined.g.vcf.gz"
    
    # Build -V arguments
    v_args = " \\\n".join([f"    -V {gvcf}" for _, gvcf in sorted_gvcfs])
    
    script_content = f"""#!/bin/bash
#$ -N joint_call_alt
#$ -o {LOGS_DIR}/joint_call_alt.out
#$ -e {LOGS_DIR}/joint_call_alt.err
#$ -cwd
#$ -V
#$ -pe smp 8
#$ -l h_vmem=32G

echo "=========================================="
echo "Step 2 (Alternative): CombineGVCFs + GenotypeGVCFs"
echo "=========================================="
echo "Start time: $(date)"

TMP_DIR="{WORK_DIR}/tmp_jointcall"
mkdir -p $TMP_DIR

# ============================================================
# Step 1: CombineGVCFs
# ============================================================
echo ""
echo "[1/3] Running CombineGVCFs..."

{GATK} --java-options "-Xmx24g -Xms4g" CombineGVCFs \\
    -R {REF} \\
{v_args} \\
    -O {combined_gvcf} \\
    --tmp-dir $TMP_DIR

if [ $? -ne 0 ]; then
    echo "ERROR: CombineGVCFs failed!"
    exit 1
fi

# ============================================================
# Step 2: GenotypeGVCFs
# ============================================================
echo ""
echo "[2/3] Running GenotypeGVCFs..."

{GATK} --java-options "-Xmx24g -Xms4g" GenotypeGVCFs \\
    -R {REF} \\
    -V {combined_gvcf} \\
    -O {OUTPUT_VCF} \\
    --include-non-variant-sites \\
    --tmp-dir $TMP_DIR

if [ $? -ne 0 ]; then
    echo "ERROR: GenotypeGVCFs failed!"
    exit 1
fi

# ============================================================
# Step 3: Index
# ============================================================
echo ""
echo "[3/3] Indexing output VCF..."

{GATK} IndexFeatureFile -I {OUTPUT_VCF}

# Summary
echo ""
echo "=========================================="
echo "Joint Calling Complete!"
echo "=========================================="
echo "Output VCF: {OUTPUT_VCF}"

bcftools stats {OUTPUT_VCF} | head -30

echo ""
echo "Samples in VCF (in order):"
bcftools query -l {OUTPUT_VCF}

echo ""
echo "End time: $(date)"

rm -rf $TMP_DIR
"""
    
    script_path = SCRIPTS_DIR / "02_joint_genotyping_alt.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    return script_path


def main():
    print("=" * 70)
    print("Step 2: Joint Genotyping Pipeline Setup")
    print("=" * 70)
    
    # Check prerequisites
    print("\n[1] Checking prerequisites...")
    
    if not REF.exists():
        print(f"  ERROR: Reference not found: {REF}")
        return
    
    if not MARKERS_BED.exists():
        print(f"  ERROR: Markers BED not found: {MARKERS_BED}")
        return
    
    # Find GVCFs
    print("\n[2] Finding GVCF files...")
    sorted_gvcfs = find_gvcfs()
    
    if not sorted_gvcfs:
        print(f"  ERROR: No GVCF files found in {GVCF_DIR}")
        print("  Please run Step 1 (HaplotypeCaller) first!")
        return
    
    print(f"  Found {len(sorted_gvcfs)} GVCF files (sorted by family order):")
    for sample_name, gvcf_path in sorted_gvcfs:
        status = "✓" if gvcf_path.exists() else "✗"
        print(f"    {status} {sample_name}: {gvcf_path.name}")
    
    # Create sample map
    print("\n[3] Creating sample map...")
    sample_map = create_sample_map(sorted_gvcfs)
    print(f"  Created: {sample_map}")
    
    # Create interval list
    print("\n[4] Creating interval list...")
    interval_list = create_interval_list()
    print(f"  Created: {interval_list}")
    
    # Create scripts
    print("\n[5] Creating joint calling scripts...")
    
    main_script = create_jointcall_script(sample_map, interval_list, sorted_gvcfs)
    print(f"  Main script (GenomicsDBImport): {main_script}")
    
    alt_script = create_alternative_script(sorted_gvcfs)
    print(f"  Alternative script (CombineGVCFs): {alt_script}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Setup Complete!")
    print("=" * 70)
    print(f"\n  Output VCF will be: {OUTPUT_VCF}")
    print(f"\n  To run joint calling:")
    print(f"    qsub {main_script}")
    print(f"\n  Or alternative method:")
    print(f"    qsub {alt_script}")
    print(f"\n  Note: Run this AFTER all GVCF jobs complete!")
    print(f"        Check with: qstat | grep gvcf")


if __name__ == "__main__":
    main()