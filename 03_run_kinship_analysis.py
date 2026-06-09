#!/usr/bin/env python3
"""
Step 3: Kinship analysis
========================
Extract marker-set VCF subsets, convert them to PLINK binary files, and run
PLINK IBS/IBD plus KING kinship locally. Outputs are written under
06_kinship_analysis by default.
"""

import argparse
import subprocess
import os
import json
import textwrap
from pathlib import Path

HOME = Path.home()
DEFAULT_WORK_DIR  = "/mnt/d/Research/20251031_wgrs"
DEFAULT_JOINT_VCF = "/mnt/d/Research/20251031_wgrs/05_jointcall/joint_called.allsites.vcf.gz"
DEFAULT_ANALYSIS_DIR = "/mnt/d/Research/20251031_wgrs/06_kinship_analysis"

ALL_FAMILIES = [1, 2, 4, 5, 6, 9, 10, 14, 15, 18]
MEMBERS = ['1A', '2B', '3a', '4b', '5c', '6D', '7E', '8d', '9e', '10f']


def load_config(config_path):
    """Load marker config from YAML or JSON.
    YAML format:
        markers:
          NFS_36K: /path/to/NFS_36K.bed
          Custom_Panel: /path/to/custom.bed
    JSON format:
        {"markers": {"NFS_36K": "/path/to/NFS_36K.bed"}}
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    suffix = config_path.suffix.lower()
    with open(config_path, 'r') as f:
        if suffix in ('.yaml', '.yml'):
            try:
                import yaml
                config = yaml.safe_load(f)
            except ImportError:
                config = _parse_simple_yaml(f)
        elif suffix == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {suffix}")
    if 'markers' not in config:
        raise ValueError("Config must have 'markers' key")
    marker_sets = {}
    for name, bed_path in config['markers'].items():
        p = Path(bed_path)
        if not p.is_absolute():
            p = config_path.parent / p
        if not p.exists():
            print(f"  WARNING: BED not found for {name}: {p}")
            continue
        marker_sets[name] = p
    return marker_sets


def _parse_simple_yaml(file_obj):
    """Minimal YAML parser (no PyYAML dependency)."""
    file_obj.seek(0)
    config = {}
    current_section = None
    for line in file_obj:
        line = line.rstrip()
        if not line or line.startswith('#'):
            continue
        stripped = line.lstrip()
        indent = len(line) - len(stripped)
        if stripped.endswith(':') and indent == 0:
            current_section = stripped[:-1].strip()
            config[current_section] = {}
        elif current_section and ':' in stripped:
            key, val = stripped.split(':', 1)
            config[current_section][key.strip()] = val.strip()
    return config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Step 3: run kinship analysis locally (VCF subset -> PLINK -> KING)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python 03_run_kinship_analysis.py --families all --config markers.yaml
          python 03_run_kinship_analysis.py --families 1,2,5,6 --36k beds/NFS_36K.bed --12k beds/NFS_12K.bed
        """))
    parser.add_argument('--families', required=True,
                        help='"all" or comma-separated (e.g. "1,2,5,6")')
    parser.add_argument('--config', dest='config_file',
                        help='YAML/JSON config with marker names and BED paths')
    mg = parser.add_argument_group('Marker sets (CLI mode)')
    mg.add_argument('--36k', dest='bed_36k', help='BED for NFS_36K')
    mg.add_argument('--24k', dest='bed_24k', help='BED for NFS_24K')
    mg.add_argument('--12k', dest='bed_12k', help='BED for NFS_12K')
    mg.add_argument('--6k',  dest='bed_6k',  help='BED for NFS_6K')
    mg.add_argument('--kintelligence', dest='bed_kintelligence', help='BED for Kintelligence')
    mg.add_argument('--qiaseq', dest='bed_qiaseq', help='BED for QIAseq')
    parser.add_argument('--joint-vcf', default=str(DEFAULT_JOINT_VCF))
    parser.add_argument('--analysis-dir', '--outdir', dest='analysis_dir', default=str(DEFAULT_ANALYSIS_DIR),
                        help='Directory for Step 3 outputs (default: 06_kinship_analysis)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--threads', type=int, default=4)

    args = parser.parse_args()
    if args.families.lower() == 'all':
        args.family_list = ALL_FAMILIES
    else:
        args.family_list = [int(f.strip()) for f in args.families.split(',')]

    args.marker_sets = {}
    if args.config_file:
        args.marker_sets = load_config(args.config_file)
    else:
        cli = {'NFS_36K': args.bed_36k, 'NFS_24K': args.bed_24k,
               'NFS_12K': args.bed_12k, 'NFS_6K': args.bed_6k,
               'Kintelligence': args.bed_kintelligence, 'QIAseq': args.bed_qiaseq}
        for name, path in cli.items():
            if path:
                p = Path(path)
                if not p.exists():
                    parser.error(f"BED not found: {path}")
                args.marker_sets[name] = p
    if not args.marker_sets:
        parser.error("Marker sets required (--config or --36k/--24k/...)")
    args.marker_list = list(args.marker_sets.keys())
    return args

def get_sample_list(families):
    samples = []
    for fam in sorted(families):
        for member in MEMBERS:
            samples.append(f"2024-{fam:03d}-{member}")
    return samples


# ============================================================
# Step 3: Kinship Analysis (VCF extract -> PLINK -> KING)
# ============================================================
def step3_kinship_analysis(args):
    print("\n" + "=" * 70)
    print("STEP 3: Kinship Analysis")
    print("=" * 70)
    analysis_dir = Path(args.analysis_dir)
    vcf_dir, plink_dir = analysis_dir / "vcf_subsets", analysis_dir / "plink_files"
    results_dir, scripts_dir = analysis_dir / "results", analysis_dir / "scripts"
    for d in [vcf_dir, plink_dir, results_dir, scripts_dir]:
        d.mkdir(parents=True, exist_ok=True)
    joint_vcf = Path(args.joint_vcf)
    samples = get_sample_list(args.family_list)
    sample_file = analysis_dir / "selected_samples.txt"
    with open(sample_file, 'w') as f:
        for s in samples:
            f.write(s + '\n')
    print(f"  Samples: {len(samples)} from families {args.family_list}")
    job_names = []
    for name, bed_path in args.marker_sets.items():
        vcf_out = vcf_dir / f"{name}.vcf.gz"
        plink_prefix = plink_dir / name
        results_prefix = results_dir / name
        script_content = f"""#!/bin/bash
set -euo pipefail

echo "========================================"
echo "Kinship Analysis: {name}"
echo "========================================"
echo "Start: $(date)"
echo "Samples: {len(samples)}"
echo "BED: {bed_path}"

echo ""
echo "[1/4] Extracting VCF subset..."
bcftools view -S {sample_file} -R {bed_path} {joint_vcf} -Oz -o {vcf_out}
tabix -p vcf {vcf_out}
TOTAL=$(bcftools view -H {vcf_out} | wc -l)
SNPS=$(bcftools view -H -v snps {vcf_out} | wc -l)
echo "  Total sites: $TOTAL"
echo "  SNP sites: $SNPS"

echo ""
echo "[2/4] Converting to PLINK format..."
plink --vcf {vcf_out} \\
      --make-bed \\
      --out {plink_prefix} \\
      --allow-extra-chr \\
      --double-id \\
      --set-missing-var-ids @:#:\\$1:\\$2 \\
      --vcf-half-call m \\
      2>&1 | tail -5
echo "  Variants: $(wc -l < {plink_prefix}.bim)"
echo "  Samples:  $(wc -l < {plink_prefix}.fam)"

echo ""
echo "[3/4] Calculating IBS/IBD..."
plink --bfile {plink_prefix} \\
      --genome \\
      --out {results_prefix}_plink \\
      --allow-extra-chr \\
      2>&1 | tail -3
echo "  Pairs: $(tail -n +2 {results_prefix}_plink.genome | wc -l)"

echo ""
echo "[4/4] Calculating Kinship (KING)..."
king -b {plink_prefix}.bed \\
     --kinship \\
     --prefix {results_prefix}_king \\
     2>&1 | tail -5
if [ -f {results_prefix}_king.kin0 ]; then
    echo "  Pairs: $(tail -n +2 {results_prefix}_king.kin0 | wc -l)"
else
    echo "  WARNING: KING .kin0 not found"
fi
echo ""
echo "Completed: {name}"
echo "End: $(date)"
"""
        script_path = scripts_dir / f"kin_{name}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        os.chmod(script_path, 0o755)
        job_names.append(f"kin_{name}")
        print(f"  Running {name} locally...")
        if args.dry_run:
            print(f"  [DRY-RUN] bash {script_path}")
            continue
        ret = subprocess.run(["bash", str(script_path)])
        if ret.returncode != 0:
            print(f"  !! ERROR running {name}")
            return None
    return job_names


def main():
    args = parse_args()
    print("=" * 70)
    print("STEP 3: KINSHIP ANALYSIS (LOCAL)")
    print("=" * 70)
    print(f"  Families: {args.family_list}")
    print(f"  Marker sets: {args.marker_list}")
    print(f"  Joint VCF: {args.joint_vcf}")
    print(f"  Step 3 output: {args.analysis_dir}")
    if args.dry_run:
        print("  *** DRY RUN ***")
    if step3_kinship_analysis(args) is None:
        raise SystemExit(1)
    print("\nSTEP 3 COMPLETE")


if __name__ == "__main__":
    main()
