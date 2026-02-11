#!/usr/bin/env python3
"""
Kinship Analysis Unified Pipeline (v2)
========================================
Single command to run the entire post-joint-calling pipeline:
  Step 3: Extract VCF subsets -> PLINK IBS/IBD -> KING kinship
  Step 4: Generate ground truth from PED file
  Step 5: Evaluate results and generate ALL figures + report

Config file support (recommended):
  python run_pipeline.py --families all --config markers.yaml

CLI mode:
  python run_pipeline.py --families all --36k beds/NFS_36K.bed --12k beds/NFS_12K.bed

Resume:
  python run_pipeline.py --families all --config markers.yaml --start-from 4
"""

import argparse
import subprocess
import os
import sys
import json
import textwrap
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Default paths
# ============================================================
HOME = Path.home()
DEFAULT_WORK_DIR  = HOME / "kinship/Analysis/20251031_wgrs"
DEFAULT_JOINT_VCF = DEFAULT_WORK_DIR / "05_jointcall/joint_called.allsites.vcf.gz"
DEFAULT_PED_FILE  = DEFAULT_WORK_DIR / "full_pedigree.ped"
DEFAULT_OUT_DIR   = DEFAULT_WORK_DIR / "06_kinship_analysis"

ALL_FAMILIES = [1, 2, 4, 5, 6, 9, 10, 14, 15, 18]
MEMBERS = ['1A', '2B', '3a', '4b', '5c', '6D', '7E', '8d', '9e', '10f']

# ============================================================
# Plotting Constants
# ============================================================
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.facecolor'] = 'white'

MARKER_COLORS = {
    'NFS_36K': '#1a5276', 'NFS_24K': '#2874a6', 'NFS_20K': '#3498db',
    'NFS_12K': '#e74c3c', 'NFS_6K': '#9b59b6',
    'Kintelligence': '#27ae60', 'QIAseq': '#f39c12'
}
DEGREE_COLORS = {
    0: '#95a5a6', 1: '#c0392b', 2: '#e74c3c', 3: '#e67e22',
    4: '#f1c40f', 5: '#2ecc71', 6: '#3498db', 7: '#9b59b6'
}
RELATIONSHIP_COLORS = {
    'Parent-Child': '#c0392b', 'Sibling': '#e74c3c',
    'Grandparent-Grandchild': '#d35400', 'Uncle-Nephew': '#e67e22',
    'Cousin': '#f39c12', 'Grand-Uncle-Nephew': '#27ae60',
    'Cousin-Once-Removed': '#2ecc71', 'Second-Cousin': '#3498db',
    'Spouse': '#7f8c8d', 'Unrelated': '#95a5a6',
}
RELATIONSHIP_ORDER = [
    'Parent-Child', 'Sibling', 'Grandparent-Grandchild', 'Uncle-Nephew',
    'Cousin', 'Grand-Uncle-Nephew', 'Cousin-Once-Removed', 'Second-Cousin',
    'Spouse', 'Unrelated'
]
RELATIONSHIP_LABELS = {
    'Parent-Child': 'Parent-Child\n(1촌)',
    'Sibling': 'Sibling\n(2촌)',
    'Grandparent-Grandchild': 'Grandparent\n(2촌)',
    'Uncle-Nephew': 'Uncle-Nephew\n(3촌)',
    'Cousin': 'Cousin\n(4촌)',
    'Grand-Uncle-Nephew': 'Grand-Uncle\n(4촌)',
    'Cousin-Once-Removed': 'Cousin-1R\n(5촌)',
    'Second-Cousin': '2nd-Cousin\n(6촌)',
    'Spouse': 'Spouse\n(0촌)',
    'Unrelated': 'Unrelated\n(0촌)'
}

# ============================================================
# Config file loading
# ============================================================
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
    _auto_colors = ['#16a085', '#8e44ad', '#2c3e50', '#d35400',
                    '#c0392b', '#7f8c8d', '#1abc9c', '#e74c3c']
    for name, bed_path in config['markers'].items():
        p = Path(bed_path)
        if not p.is_absolute():
            p = config_path.parent / p
        if not p.exists():
            print(f"  WARNING: BED not found for {name}: {p}")
            continue
        marker_sets[name] = p
        if name not in MARKER_COLORS:
            idx = len(MARKER_COLORS) % len(_auto_colors)
            MARKER_COLORS[name] = _auto_colors[idx]
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


# ============================================================
# Argument Parsing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Kinship Analysis Unified Pipeline (v2)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python run_pipeline.py --families all --config markers.yaml
          python run_pipeline.py --families all --36k beds/NFS_36K.bed --12k beds/NFS_12K.bed
          python run_pipeline.py --families 1,2,5,6 --config markers.yaml --start-from 4
        """))
    parser.add_argument('--families', required=True,
                        help='"all" or comma-separated (e.g. "1,2,5,6")')
    parser.add_argument('--config', dest='config_file',
                        help='YAML/JSON config with marker names and BED paths')
    mg = parser.add_argument_group('Marker sets (CLI mode)')
    mg.add_argument('--36k', dest='bed_36k', help='BED for NFS_36K')
    mg.add_argument('--24k', dest='bed_24k', help='BED for NFS_24K')
    mg.add_argument('--20k', dest='bed_20k', help='BED for NFS_20K')
    mg.add_argument('--12k', dest='bed_12k', help='BED for NFS_12K')
    mg.add_argument('--6k',  dest='bed_6k',  help='BED for NFS_6K')
    mg.add_argument('--kintelligence', dest='bed_kintelligence', help='BED for Kintelligence')
    mg.add_argument('--qiaseq', dest='bed_qiaseq', help='BED for QIAseq')
    pg = parser.add_argument_group('Paths')
    pg.add_argument('--joint-vcf', default=str(DEFAULT_JOINT_VCF))
    pg.add_argument('--ped', default=str(DEFAULT_PED_FILE))
    pg.add_argument('--outdir', default=str(DEFAULT_OUT_DIR))
    eg = parser.add_argument_group('Execution')
    eg.add_argument('--run-mode', choices=['qsub', 'local'], default='qsub')
    eg.add_argument('--start-from', type=int, choices=[3, 4, 5], default=3)
    eg.add_argument('--dry-run', action='store_true')
    eg.add_argument('--threads', type=int, default=4)

    args = parser.parse_args()
    if args.families.lower() == 'all':
        args.family_list = ALL_FAMILIES
    else:
        args.family_list = [int(f.strip()) for f in args.families.split(',')]

    # Load marker sets: config > CLI
    args.marker_sets = {}
    if args.config_file:
        args.marker_sets = load_config(args.config_file)
    else:
        cli = {'NFS_36K': args.bed_36k, 'NFS_24K': args.bed_24k, 'NFS_20K': args.bed_20k,
               'NFS_12K': args.bed_12k, 'NFS_6K': args.bed_6k,
               'Kintelligence': args.bed_kintelligence, 'QIAseq': args.bed_qiaseq}
        for name, path in cli.items():
            if path:
                p = Path(path)
                if not p.exists():
                    parser.error(f"BED not found: {path}")
                args.marker_sets[name] = p
    if not args.marker_sets and args.start_from <= 3:
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
    outdir = Path(args.outdir)
    vcf_dir, plink_dir = outdir / "vcf_subsets", outdir / "plink_files"
    results_dir, scripts_dir, logs_dir = outdir / "results", outdir / "scripts", outdir / "logs"
    for d in [vcf_dir, plink_dir, results_dir, scripts_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    joint_vcf = Path(args.joint_vcf)
    samples = get_sample_list(args.family_list)
    sample_file = outdir / "selected_samples.txt"
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
#$ -N kin_{name}
#$ -o {logs_dir}/kin_{name}.out
#$ -e {logs_dir}/kin_{name}.err
#$ -cwd
#$ -V
#$ -pe smp {args.threads}

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
        if args.run_mode == 'qsub':
            if not args.dry_run:
                result = subprocess.run(f"qsub {script_path}", shell=True, capture_output=True, text=True)
                print(f"  Submitted: {name} -> {result.stdout.strip()}")
            else:
                print(f"  [DRY-RUN] qsub {script_path}")
        else:
            print(f"  Running {name} locally...")
            if not args.dry_run:
                ret = os.system(f"bash {script_path}")
                if ret != 0:
                    print(f"  !! ERROR running {name}")
                    return None
    return job_names


# ============================================================
# Step 4: Ground Truth Generation (FIXED)
# ============================================================
class FamilyTree:
    """Pedigree relationship inference with corrected kinship coefficients.
    Fixes: Grand-Uncle-Nephew=4촌, LCA path multiplier in Wright's formula,
    GP/GM virtual samples excluded."""
    def __init__(self, family_id):
        self.family_id = family_id
        self.members = {}
        self.children = defaultdict(list)

    def add_member(self, member_id, father_id, mother_id, sex):
        self.members[member_id] = {
            'father': father_id if father_id else None,
            'mother': mother_id if mother_id else None, 'sex': sex}
        if father_id:
            self.children[father_id].append(member_id)
        if mother_id:
            self.children[mother_id].append(member_id)

    def get_parents(self, member_id):
        if member_id not in self.members:
            return []
        m = self.members[member_id]
        return [p for p in [m['father'], m['mother']] if p]

    def get_all_ancestors(self, member_id, max_depth=10):
        ancestors = {}
        def _trace(cid, depth):
            if depth > max_depth:
                return
            for pid in self.get_parents(cid):
                if pid not in ancestors or ancestors[pid] > depth:
                    ancestors[pid] = depth
                    _trace(pid, depth + 1)
        _trace(member_id, 1)
        return ancestors

    def find_all_lcas(self, id1, id2):
        if id1 == id2:
            return [(id1, 0, 0)]
        a1 = self.get_all_ancestors(id1); a1[id1] = 0
        a2 = self.get_all_ancestors(id2); a2[id2] = 0
        common = set(a1) & set(a2)
        if not common:
            return []
        min_total = min(a1[a] + a2[a] for a in common)
        return [(a, a1[a], a2[a]) for a in common if a1[a] + a2[a] == min_total]

    def get_relationship(self, id1, id2):
        """Returns (name, chon, wright_kinship). phi = n_paths * (1/2)^(d1+d2+1)"""
        if id1 == id2:
            return ("Self", 0, 0.5)
        common_children = set(self.children.get(id1, [])) & set(self.children.get(id2, []))
        lcas = self.find_all_lcas(id1, id2)
        if not lcas:
            return ("Spouse", 0, 0.0) if common_children else ("Unrelated", 0, 0.0)
        _, d1, d2 = lcas[0]
        n_paths = len(lcas)
        chon = d1 + d2
        kinship = n_paths * (0.5) ** (d1 + d2 + 1)
        if d1 == 0 or d2 == 0:
            names = {1: "Parent-Child", 2: "Grandparent-Grandchild", 3: "Great-Grandparent"}
            rel = names.get(chon, f"Direct-{chon}촌")
        elif (d1, d2) == (1, 1):
            rel = "Sibling"
        elif sorted([d1, d2]) == [1, 2]:
            rel = "Uncle-Nephew"
        elif (d1, d2) == (2, 2):
            rel = "Cousin"
        elif sorted([d1, d2]) == [1, 3]:
            rel = "Grand-Uncle-Nephew"
        elif sorted([d1, d2]) == [2, 3]:
            rel = "Cousin-Once-Removed"
        elif (d1, d2) == (3, 3):
            rel = "Second-Cousin"
        elif sorted([d1, d2]) == [1, 4]:
            rel = "Great-Grand-Uncle"
        elif sorted([d1, d2]) == [2, 4]:
            rel = "Cousin-Twice-Removed"
        else:
            rel = f"Distant-{chon}촌"
        return (rel, chon, kinship)


def step4_ground_truth(args):
    print("\n" + "=" * 70)
    print("STEP 4: Ground Truth Generation")
    print("=" * 70)
    ped_path = Path(args.ped)
    if not ped_path.exists():
        print(f"  ERROR: PED file not found: {ped_path}")
        return None
    families = {}
    with open(ped_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split('\t') if '\t' in line else line.split()
            if len(fields) < 5:
                continue
            fam_id, ind_id = fields[0], fields[1]
            fam_num = int(fam_id.replace('FAM', '').replace('fam', ''))
            if fam_num not in args.family_list:
                continue
            father = fields[2] if fields[2] and fields[2] != '0' else None
            mother = fields[3] if fields[3] and fields[3] != '0' else None
            sex = int(fields[4]) if fields[4].isdigit() else 0
            if fam_id not in families:
                families[fam_id] = FamilyTree(fam_id)
            families[fam_id].add_member(ind_id, father, mother, sex)
    print(f"  Families loaded: {len(families)}")
    for fid in sorted(families):
        n_seq = sum(1 for m in families[fid].members if 'GP' not in m and 'GM' not in m)
        print(f"    {fid}: {n_seq} sequenced members")
    results = []
    for fam_id, tree in families.items():
        seq_members = [m for m in tree.members if 'GP' not in m and 'GM' not in m]
        for id1, id2 in combinations(seq_members, 2):
            rel, chon, ek = tree.get_relationship(id1, id2)
            p1, p2 = id1.split('-'), id2.split('-')
            results.append({
                'Sample1': id1, 'Sample2': id2,
                'Family1': int(p1[1]), 'Family2': int(p2[1]),
                'Member1': p1[2], 'Member2': p2[2],
                'Relationship': rel, 'Degree': chon,
                'Expected_Kinship': ek, 'Same_Family': True, 'Is_Related': chon > 0})
    fam_ids = list(families.keys())
    for i, f1 in enumerate(fam_ids):
        for f2 in fam_ids[i+1:]:
            m1s = [m for m in families[f1].members if 'GP' not in m and 'GM' not in m]
            m2s = [m for m in families[f2].members if 'GP' not in m and 'GM' not in m]
            for id1 in m1s:
                for id2 in m2s:
                    p1, p2 = id1.split('-'), id2.split('-')
                    results.append({
                        'Sample1': id1, 'Sample2': id2,
                        'Family1': int(p1[1]), 'Family2': int(p2[1]),
                        'Member1': p1[2], 'Member2': p2[2],
                        'Relationship': 'Unrelated', 'Degree': 0,
                        'Expected_Kinship': 0.0, 'Same_Family': False, 'Is_Related': False})
    gt_df = pd.DataFrame(results)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gt_path = outdir / "family_relationships.csv"
    gt_df.to_csv(gt_path, index=False)
    n_related = len(gt_df[gt_df['Is_Related']])
    print(f"\n  Total pairs: {len(gt_df):,}")
    print(f"  Related pairs: {n_related}")
    print(f"  Saved: {gt_path}")
    print(f"\n  {'Degree':>6} {'Count':>6}  Relationships")
    print("  " + "-" * 50)
    for deg in sorted(gt_df['Degree'].unique()):
        sub = gt_df[gt_df['Degree'] == deg]
        rels = sub.groupby('Relationship').size()
        rel_str = ", ".join(f"{r}({n})" for r, n in rels.items())
        print(f"  {deg:>4}촌 {len(sub):>6}  {rel_str}")
    return gt_df


# ============================================================
# Step 5: Data Loading
# ============================================================
def load_plink_genome(filepath):
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath, delim_whitespace=True)
    df['Sample1'] = df['IID1'].astype(str)
    df['Sample2'] = df['IID2'].astype(str)
    df['pair'] = df.apply(lambda r: tuple(sorted([r['Sample1'], r['Sample2']])), axis=1)
    return df[['pair', 'Sample1', 'Sample2', 'PI_HAT', 'DST', 'Z0', 'Z1', 'Z2']]

def load_king_kinship(filepath):
    for ext in ['.kin0', '.kin']:
        test_path = filepath.with_suffix(ext)
        if test_path.exists():
            filepath = test_path
            break
    else:
        return None
    df = pd.read_csv(filepath, sep='\t')
    if 'ID1' in df.columns:
        df['Sample1'] = df['ID1'].astype(str)
        df['Sample2'] = df['ID2'].astype(str)
    elif 'IID1' in df.columns:
        df['Sample1'] = df['IID1'].astype(str)
        df['Sample2'] = df['IID2'].astype(str)
    df['pair'] = df.apply(lambda r: tuple(sorted([r['Sample1'], r['Sample2']])), axis=1)
    return df[['pair', 'Sample1', 'Sample2', 'Kinship']]

def merge_results(ground_truth, marker_set, results_dir):
    plink_file = results_dir / f"{marker_set}_plink.genome"
    king_file = results_dir / f"{marker_set}_king.kin0"
    plink_df = load_plink_genome(plink_file)
    king_df = load_king_kinship(king_file)
    gt = ground_truth.copy()
    gt['pair'] = gt.apply(lambda r: tuple(sorted([str(r['Sample1']), str(r['Sample2'])])), axis=1)
    if plink_df is not None:
        plink_df = plink_df.rename(columns={'PI_HAT': 'IBD', 'DST': 'IBS'})
        gt = gt.merge(plink_df[['pair', 'IBD', 'IBS', 'Z0', 'Z1', 'Z2']], on='pair', how='left')
    else:
        for col in ['IBD', 'IBS', 'Z0', 'Z1', 'Z2']:
            gt[col] = np.nan
    if king_df is not None:
        gt = gt.merge(king_df[['pair', 'Kinship']], on='pair', how='left')
    else:
        gt['Kinship'] = np.nan
    gt['Marker_Set'] = marker_set
    return gt


# ============================================================
# Plotting Functions
# ============================================================
def plot_boxplot_by_degree_all(all_df, marker_set, output_path):
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    if len(df) == 0: return
    def get_label(row):
        if row['Same_Family']:
            return 'Spouse/InLaw\n(0촌)' if row['Degree'] == 0 else f"{row['Degree']}촌"
        return 'Between-Fam\n(Unrel)'
    df['DL'] = df.apply(get_label, axis=1)
    order = ['1촌','2촌','3촌','4촌','5촌','6촌','Spouse/InLaw\n(0촌)','Between-Fam\n(Unrel)']
    avail = [o for o in order if o in df['DL'].values]
    pal = []
    for o in avail:
        for d in range(1, 7):
            if f'{d}촌' in o:
                pal.append(DEGREE_COLORS[d]); break
        else:
            pal.append(DEGREE_COLORS[0])
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    for ax, m in zip(axes, ['IBS', 'IBD', 'Kinship']):
        data = df.dropna(subset=[m])
        if len(data) == 0:
            ax.set_title(f'{m} (No Data)'); continue
        sns.boxplot(data=data, x='DL', y=m, order=avail, palette=pal, ax=ax,
                    width=0.6, linewidth=1.5, flierprops={'marker':'o','markersize':3,'alpha':0.3})
        sns.stripplot(data=data, x='DL', y=m, order=avail, color='black', size=2, alpha=0.3, ax=ax, jitter=True)
        for i, deg in enumerate(avail):
            n = len(data[data['DL'] == deg])
            ymin = data[m].min() - (data[m].max() - data[m].min()) * 0.1
            ax.annotate(f'n={n}', xy=(i, ymin), ha='center', va='top', fontsize=9, color='gray', style='italic')
        ax.set_xlabel('Degree', fontsize=12); ax.set_ylabel(m, fontsize=12)
        ax.set_title(f'{m}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45); ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f'{marker_set} - Distribution by Degree', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"    Saved: {output_path.name}")

def plot_boxplot_by_relationship(all_df, marker_set, output_path):
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    if len(df) == 0: return
    rel_order = [r for r in RELATIONSHIP_ORDER if r in df['Relationship'].values]
    if not rel_order: return
    df['RL'] = df['Relationship'].map(lambda x: RELATIONSHIP_LABELS.get(x, x))
    lo = [RELATIONSHIP_LABELS.get(r, r) for r in rel_order]
    pal = [RELATIONSHIP_COLORS.get(r, '#95a5a6') for r in rel_order]
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    for ax, m in zip(axes, ['IBS', 'IBD', 'Kinship']):
        data = df.dropna(subset=[m])
        if len(data) == 0: continue
        sns.boxplot(data=data, x='RL', y=m, order=lo, palette=pal, ax=ax, width=0.6, linewidth=1.5)
        sns.stripplot(data=data, x='RL', y=m, order=lo, color='black', size=2, alpha=0.3, ax=ax, jitter=True)
        for i, label in enumerate(lo):
            n = len(data[data['RL'] == label])
            ymin = data[m].min() - (data[m].max() - data[m].min()) * 0.08
            ax.annotate(f'n={n}', xy=(i, ymin), ha='center', va='top', fontsize=8, color='gray', style='italic')
        ax.set_xlabel('Relationship', fontsize=12); ax.set_ylabel(m, fontsize=12)
        ax.set_title(f'{m}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=9); ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f'{marker_set} - Distribution by Relationship', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"    Saved: {output_path.name}")

def plot_violin_by_relationship(all_df, marker_set, output_path):
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    if len(df) == 0: return
    rel_order = [r for r in RELATIONSHIP_ORDER if r in df['Relationship'].values]
    if not rel_order: return
    df['RL'] = df['Relationship'].map(lambda x: RELATIONSHIP_LABELS.get(x, x))
    lo = [RELATIONSHIP_LABELS.get(r, r) for r in rel_order]
    pal = [RELATIONSHIP_COLORS.get(r, '#95a5a6') for r in rel_order]
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    for ax, m in zip(axes, ['IBS', 'IBD', 'Kinship']):
        data = df.dropna(subset=[m])
        if len(data) == 0: continue
        sns.violinplot(data=data, x='RL', y=m, order=lo, palette=pal, ax=ax, inner='box', linewidth=1)
        ax.set_xlabel('Relationship', fontsize=12); ax.set_ylabel(m, fontsize=12)
        ax.set_title(f'{m}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=9); ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f'{marker_set} - Violin by Relationship', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()

def plot_heatmap_standard(all_df, marker_set, metric, output_path):
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    if len(df) == 0 or df[metric].isna().all(): return
    samples = sorted(set(df['Sample1'].tolist() + df['Sample2'].tolist()))
    samples = [s for s in samples if 'GP' not in s and 'GM' not in s]
    n = len(samples)
    if n == 0: return
    matrix = np.full((n, n), np.nan)
    si = {s: i for i, s in enumerate(samples)}
    for _, row in df.iterrows():
        s1, s2 = str(row['Sample1']), str(row['Sample2'])
        i, j = si.get(s1), si.get(s2)
        val = row[metric]
        if i is not None and j is not None and pd.notna(val):
            matrix[i, j] = val; matrix[j, i] = val
    np.fill_diagonal(matrix, 0.5 if metric == 'Kinship' else 1.0)
    figsize = max(14, n * 0.25)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.9))
    vranges = {'IBS': (0.55, 0.85), 'IBD': (0, 0.6), 'Kinship': (-0.05, 0.3)}
    vmin, vmax = vranges.get(metric, (0, 1))
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    sns.heatmap(matrix, mask=mask, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                square=True, linewidths=0.2, linecolor='white',
                cbar_kws={'shrink': 0.6, 'label': metric}, ax=ax)
    labels = [f"{s.split('-')[1]}-{s.split('-')[2]}" if len(s.split('-')) >= 3 else s for s in samples]
    fs = max(4, min(8, 120 // n))
    ax.set_xticks(np.arange(n) + 0.5); ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=fs)
    ax.set_yticklabels(labels, rotation=0, ha='right', fontsize=fs)
    ax.set_title(f'{marker_set} - {metric}', fontsize=14, fontweight='bold')
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"    Saved: {output_path.name}")

def plot_heatmap_within_family(all_df, marker_set, metric, family, output_path):
    df = all_df[(all_df['Marker_Set'] == marker_set) &
                (all_df['Family1'] == family) & (all_df['Family2'] == family)].copy()
    if len(df) == 0: return
    samples = sorted(set(df['Sample1'].tolist() + df['Sample2'].tolist()))
    samples = [s for s in samples if 'GP' not in s and 'GM' not in s]
    n = len(samples)
    if n == 0: return
    matrix = np.full((n, n), np.nan)
    si = {s: i for i, s in enumerate(samples)}
    for _, row in df.iterrows():
        s1, s2 = str(row['Sample1']), str(row['Sample2'])
        i, j = si.get(s1), si.get(s2)
        val = row[metric]
        if i is not None and j is not None and pd.notna(val):
            matrix[i, j] = val; matrix[j, i] = val
    np.fill_diagonal(matrix, 0.5 if metric == 'Kinship' else 1.0)
    fig, ax = plt.subplots(figsize=(10, 9))
    vranges = {'IBS': (0.6, 0.85), 'IBD': (0, 0.55), 'Kinship': (-0.05, 0.3)}
    vmin, vmax = vranges.get(metric, (0, 1))
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    sns.heatmap(matrix, mask=mask, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                square=True, linewidths=0.5, linecolor='white',
                annot=True, fmt='.3f', annot_kws={'size': 9},
                cbar_kws={'shrink': 0.7, 'label': metric}, ax=ax)
    labels = [s.split('-')[-1] for s in samples]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(labels, rotation=0, fontsize=11)
    ax.set_title(f'Family {family} - {marker_set} - {metric}', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()


# ============================================================
# ROC calculation (all 13 scenarios)
# ============================================================
def calculate_roc_metrics(y_true, y_score):
    valid = ~np.isnan(y_score)
    y_true, y_score = np.array(y_true)[valid], np.array(y_score)[valid]
    if len(np.unique(y_true)) < 2 or len(y_true) == 0:
        return None, None, None, None
    try:
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, th = roc_curve(y_true, y_score)
        return auc, fpr, tpr, th
    except:
        return None, None, None, None

def calculate_all_roc_scenarios(all_df, marker_list):
    scenarios = {
        'related_vs_unrelated':       {'pos': lambda d: d['Is_Related']==True, 'neg': lambda d: d['Is_Related']==False, 'desc': 'Related vs Unrelated (All)'},
        'blood_within_vs_unrelated':  {'pos': lambda d: (d['Same_Family']==True)&(d['Degree']>0), 'neg': lambda d: d['Is_Related']==False, 'desc': 'Blood vs Unrelated'},
        'close_vs_unrelated':         {'pos': lambda d: d['Degree'].isin([1,2,3,4]), 'neg': lambda d: d['Is_Related']==False, 'desc': '1-4촌 vs Unrelated'},
        'distant_vs_unrelated':       {'pos': lambda d: d['Degree'].isin([5,6]), 'neg': lambda d: d['Is_Related']==False, 'desc': '5-6촌 vs Unrelated'},
        '1st_vs_2nd':  {'pos': lambda d: d['Degree']==1, 'neg': lambda d: d['Degree']==2, 'desc': '1촌 vs 2촌'},
        '2nd_vs_3rd':  {'pos': lambda d: d['Degree']==2, 'neg': lambda d: d['Degree']==3, 'desc': '2촌 vs 3촌'},
        '3rd_vs_4th':  {'pos': lambda d: d['Degree']==3, 'neg': lambda d: d['Degree']==4, 'desc': '3촌 vs 4촌'},
        '4th_vs_5th':  {'pos': lambda d: d['Degree']==4, 'neg': lambda d: d['Degree']==5, 'desc': '4촌 vs 5촌'},
        '5th_vs_6th':  {'pos': lambda d: d['Degree']==5, 'neg': lambda d: d['Degree']==6, 'desc': '5촌 vs 6촌'},
        '4th_vs_unrelated':  {'pos': lambda d: d['Degree']==4, 'neg': lambda d: d['Is_Related']==False, 'desc': '4촌 vs Unrelated'},
        '5th_vs_unrelated':  {'pos': lambda d: d['Degree']==5, 'neg': lambda d: d['Is_Related']==False, 'desc': '5촌 vs Unrelated'},
        '6th_vs_unrelated':  {'pos': lambda d: d['Degree']==6, 'neg': lambda d: d['Is_Related']==False, 'desc': '6촌 vs Unrelated'},
        '12345_vs_6':  {'pos': lambda d: d['Degree'].isin([1,2,3,4,5]), 'neg': lambda d: d['Degree']==6, 'desc': '1-5촌 vs 6촌'},
    }
    results = []
    for ms in marker_list:
        df = all_df[all_df['Marker_Set'] == ms]
        for sn, sd in scenarios.items():
            pm, nm = sd['pos'](df), sd['neg'](df)
            pd_, nd = df[pm], df[nm]
            if len(pd_) == 0 or len(nd) == 0: continue
            combined = pd.concat([pd_, nd])
            yt = pm[combined.index].astype(int)
            for metric in ['IBS', 'IBD', 'Kinship']:
                auc, fpr, tpr, th = calculate_roc_metrics(yt, combined[metric].values)
                opt = None
                if auc is not None and th is not None:
                    opt = th[np.argmax(tpr - fpr)]
                results.append({'Marker_Set': ms, 'Scenario': sn, 'Description': sd['desc'],
                                'Metric': metric, 'AUC': auc, 'Optimal_Threshold': opt,
                                'N_Positive': len(pd_), 'N_Negative': len(nd)})
    return pd.DataFrame(results)

def plot_roc_curves(all_df, scenario_name, pos_filter, neg_filter, title, marker_list, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ['IBS', 'IBD', 'Kinship']):
        for ms in marker_list:
            df = all_df[all_df['Marker_Set'] == ms]
            pm, nm = pos_filter(df), neg_filter(df)
            if pm.sum() == 0 or nm.sum() == 0: continue
            combined = pd.concat([df[pm], df[nm]])
            yt = pm[combined.index].astype(int)
            auc, fpr, tpr, _ = calculate_roc_metrics(yt, combined[metric].values)
            if auc is not None:
                ax.plot(fpr, tpr, label=f'{ms} ({auc:.3f})',
                        color=MARKER_COLORS.get(ms, 'gray'), linewidth=2)
        ax.plot([0,1],[0,1],'k--',alpha=0.5)
        ax.set_xlabel('FPR', fontsize=12); ax.set_ylabel('TPR', fontsize=12)
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8); ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    plt.suptitle(f'ROC: {title}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()

def plot_auc_heatmap(roc_results, metric, marker_list, output_path):
    data = roc_results[roc_results['Metric'] == metric].copy()
    if len(data) == 0: return
    pivot = data.pivot(index='Marker_Set', columns='Scenario', values='AUC')
    mo = [m for m in marker_list if m in pivot.index]
    if not mo: return
    pivot = pivot.reindex(mo).apply(pd.to_numeric, errors='coerce')
    if pivot.isna().all().all(): return
    fig, ax = plt.subplots(figsize=(16, max(4, len(mo)*0.8+2)))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                ax=ax, linewidths=0.5, cbar_kws={'label':'AUC','shrink':0.8}, annot_kws={'size':9})
    ax.set_title(f'{metric} - AUC by Scenario', fontsize=14, fontweight='bold')
    ax.set_xlabel('Scenario'); ax.set_ylabel('Marker Set')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()

def plot_adjacent_discrimination(roc_results, marker_list, output_path):
    adj = ['1st_vs_2nd','2nd_vs_3rd','3rd_vs_4th','4th_vs_5th','5th_vs_6th']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, metric in zip(axes, ['IBS', 'IBD', 'Kinship']):
        md = roc_results[(roc_results['Metric']==metric) & (roc_results['Scenario'].isin(adj))]
        for ms in marker_list:
            msd = md[md['Marker_Set']==ms]
            yv = []
            for s in adj:
                row = msd[msd['Scenario']==s]
                yv.append(row['AUC'].values[0] if len(row)>0 and pd.notna(row['AUC'].values[0]) else np.nan)
            ax.plot(range(len(adj)), yv, 'o-', color=MARKER_COLORS.get(ms,'gray'), label=ms, linewidth=2, markersize=8)
        ax.set_xticks(range(len(adj)))
        ax.set_xticklabels(['1v2','2v3','3v4','4v5','5v6'], fontsize=10)
        ax.set_xlabel('Adjacent (촌)', fontsize=12); ax.set_ylabel('AUC', fontsize=12)
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(0.4, 1.05)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    plt.suptitle('Adjacent Degree Discrimination', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()

def plot_scatter_expected_vs_observed(all_df, marker_set, output_path):
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    if len(df) == 0: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, m in zip(axes, ['IBS', 'IBD', 'Kinship']):
        data = df.dropna(subset=[m])
        for deg in sorted(data['Degree'].unique()):
            dd = data[data['Degree']==deg]
            ax.scatter(dd['Expected_Kinship'], dd[m], c=DEGREE_COLORS.get(deg,'#95a5a6'),
                       label=f"{deg}촌" if deg>0 else "Unrel", alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
        valid = data[['Expected_Kinship', m]].dropna()
        if len(valid) > 2:
            corr = valid['Expected_Kinship'].corr(valid[m])
            ax.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, fontweight='bold')
        if m == 'Kinship':
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlabel('Expected Kinship'); ax.set_ylabel(f'Observed {m}')
        ax.set_title(f'{m}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8, ncol=2); ax.grid(alpha=0.3)
    plt.suptitle(f'{marker_set} - Expected vs Observed', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()


# ============================================================
# NEW: Marker Comparison Overlay
# ============================================================
def plot_marker_comparison_overlay(all_df, marker_list, metric, output_path):
    """All markers side-by-side per degree for a given metric."""
    related = all_df[all_df['Degree'] > 0].copy()
    if len(related) == 0: return
    related['DL'] = related['Degree'].apply(lambda d: f"{d}촌")
    degrees = sorted(related['Degree'].unique())
    fig, ax = plt.subplots(figsize=(max(14, len(degrees)*2.5), 7))
    data = related.dropna(subset=[metric])
    if len(data) == 0: plt.close(); return
    sns.boxplot(data=data, x='DL', y=metric, hue='Marker_Set',
                order=[f"{d}촌" for d in degrees], hue_order=marker_list, ax=ax,
                palette={m: MARKER_COLORS.get(m,'gray') for m in marker_list}, width=0.8, linewidth=1)
    ax.set_xlabel('Degree (촌)', fontsize=13); ax.set_ylabel(metric, fontsize=13)
    ax.set_title(f'Marker Comparison - {metric}', fontsize=15, fontweight='bold')
    ax.legend(title='Marker Set', loc='upper right', fontsize=8); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"    Saved: {output_path.name}")


# ============================================================
# NEW: Per-degree Summary Statistics
# ============================================================
def generate_degree_summary_stats(all_df, marker_list, output_csv, output_plot):
    rows = []
    for ms in marker_list:
        df = all_df[all_df['Marker_Set'] == ms]
        for deg in sorted(df['Degree'].unique()):
            sub = df[df['Degree'] == deg]
            for metric in ['IBS', 'IBD', 'Kinship']:
                vals = sub[metric].dropna()
                if len(vals) == 0: continue
                rows.append({'Marker_Set': ms, 'Degree': deg, 'Metric': metric,
                             'N': len(vals), 'Mean': vals.mean(), 'Std': vals.std(),
                             'CV': vals.std()/vals.mean() if vals.mean() != 0 else np.nan,
                             'Min': vals.min(), 'Max': vals.max(), 'Median': vals.median(),
                             'Q1': vals.quantile(0.25), 'Q3': vals.quantile(0.75)})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_csv, index=False)
    print(f"    Saved: {output_csv.name}")
    kin = summary[summary['Metric'] == 'Kinship']
    if len(kin) == 0: return summary
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ms in marker_list:
        s = kin[(kin['Marker_Set']==ms) & (kin['Degree']>0)]
        if len(s)==0: continue
        axes[0].errorbar(s['Degree'], s['Mean'], yerr=s['Std'], fmt='o-',
                         color=MARKER_COLORS.get(ms,'gray'), label=ms, linewidth=1.5, capsize=3, markersize=6)
        axes[1].plot(s['Degree'], s['CV'], 'o-', color=MARKER_COLORS.get(ms,'gray'), label=ms, linewidth=1.5, markersize=6)
    axes[0].set_xlabel('Degree (촌)'); axes[0].set_ylabel('Kinship (Mean±Std)')
    axes[0].set_title('Kinship by Degree', fontweight='bold'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Degree (촌)'); axes[1].set_ylabel('CV')
    axes[1].set_title('Kinship Variability', fontweight='bold'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.suptitle('Per-Degree Summary', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(output_plot, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"    Saved: {output_plot.name}")
    return summary


# ============================================================
# NEW: Effect Size (Cohen's d) Between Adjacent Degrees
# ============================================================
def plot_effect_size_adjacent(all_df, marker_list, output_path):
    adj_pairs = [(1,2),(2,3),(3,4),(4,5),(5,6)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, metric in zip(axes, ['IBS', 'IBD', 'Kinship']):
        for ms in marker_list:
            df = all_df[all_df['Marker_Set'] == ms]
            dv = []
            for d1, d2 in adj_pairs:
                g1 = df[df['Degree']==d1][metric].dropna()
                g2 = df[df['Degree']==d2][metric].dropna()
                if len(g1) < 2 or len(g2) < 2: dv.append(np.nan); continue
                ps = np.sqrt(((len(g1)-1)*g1.std()**2 + (len(g2)-1)*g2.std()**2)/(len(g1)+len(g2)-2))
                dv.append(abs(g1.mean()-g2.mean())/ps if ps > 0 else np.nan)
            ax.plot(range(len(adj_pairs)), dv, 'o-', color=MARKER_COLORS.get(ms,'gray'), label=ms, linewidth=2, markersize=8)
        ax.set_xticks(range(len(adj_pairs)))
        ax.set_xticklabels([f'{a}v{b}' for a,b in adj_pairs], fontsize=10)
        ax.set_xlabel('Adjacent Degrees (촌)'); ax.set_ylabel("Cohen's d")
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=8); ax.grid(alpha=0.3)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.4)
        ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.4)
        ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.4)
    plt.suptitle("Effect Size (Cohen's d) Between Adjacent Degrees", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"    Saved: {output_path.name}")


# ============================================================
# NEW: Confusion Matrix at Optimal Threshold
# ============================================================
def plot_confusion_matrices(all_df, roc_results, marker_list, output_path):
    scenario, metric = 'related_vs_unrelated', 'Kinship'
    n_mk = min(len(marker_list), 4)
    if n_mk == 0: return
    fig, axes = plt.subplots(1, n_mk, figsize=(5*n_mk, 5))
    if n_mk == 1: axes = [axes]
    for idx, ms in enumerate(marker_list[:4]):
        ax = axes[idx]
        row = roc_results[(roc_results['Marker_Set']==ms) & (roc_results['Scenario']==scenario) & (roc_results['Metric']==metric)]
        if len(row)==0 or pd.isna(row['Optimal_Threshold'].values[0]):
            ax.set_title(f'{ms}\n(No threshold)'); ax.axis('off'); continue
        th = row['Optimal_Threshold'].values[0]
        df = all_df[all_df['Marker_Set']==ms].dropna(subset=[metric])
        yt = df['Is_Related'].astype(int).values
        yp = (df[metric] >= th).astype(int).values
        cm = confusion_matrix(yt, yp)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred\nUnrel','Pred\nRel'], yticklabels=['Act\nUnrel','Act\nRel'])
        ax.set_title(f'{ms}\n(th={th:.4f})', fontsize=11, fontweight='bold')
    plt.suptitle(f'Confusion Matrix: Related vs Unrelated ({metric})', fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"    Saved: {output_path.name}")


# ============================================================
# Report Generation (enhanced)
# ============================================================
def generate_report(all_df, roc_results, marker_list, report_path):
    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("KINSHIP MARKER PERFORMANCE EVALUATION REPORT\n")
        f.write("=" * 100 + "\n\n")
        sd = all_df[all_df['Marker_Set'] == marker_list[0]]
        np_ = len(sd)
        nr = len(sd[sd['Is_Related']==True])
        nw = len(sd[sd['Same_Family']==True])
        f.write("1. DATASET SUMMARY\n" + "-"*50 + "\n")
        f.write(f"  Total pairs: {np_:,}\n  Blood-related: {nr:,}\n")
        f.write(f"  Within-family: {nw:,}\n  Between-family: {np_-nw:,}\n")
        f.write(f"  Markers: {', '.join(marker_list)}\n\n")
        f.write("2. RELATIONSHIP DISTRIBUTION\n" + "-"*50 + "\n")
        for rel, cnt in sd['Relationship'].value_counts().items():
            f.write(f"  {rel:<30}: {cnt:>6}\n")
        f.write("\n3. DEGREE DISTRIBUTION\n" + "-"*50 + "\n")
        dc = sd.groupby('Degree').agg({'Sample1':'count','Expected_Kinship':'first'})
        for deg, row in dc.iterrows():
            label = f"{deg}촌" if deg > 0 else "Unrelated"
            f.write(f"  {label:<15}: {int(row['Sample1']):>6} pairs  (phi={row['Expected_Kinship']:.4f})\n")
        f.write("\n4. CLASSIFICATION PERFORMANCE (AUC)\n" + "-"*80 + "\n")
        for sn in ['related_vs_unrelated','close_vs_unrelated','distant_vs_unrelated',
                    '4th_vs_unrelated','5th_vs_unrelated','6th_vs_unrelated','12345_vs_6']:
            sdata = roc_results[roc_results['Scenario']==sn]
            if len(sdata) == 0: continue
            desc = sdata['Description'].iloc[0]
            f.write(f"\n  [{desc}]\n  {'Marker':<15} {'IBS':>10} {'IBD':>10} {'Kinship':>10}\n  " + "-"*50 + "\n")
            for mk in marker_list:
                vals = {}
                for m in ['IBS','IBD','Kinship']:
                    r = sdata[(sdata['Marker_Set']==mk) & (sdata['Metric']==m)]
                    vals[m] = f"{r['AUC'].values[0]:.4f}" if len(r)>0 and pd.notna(r['AUC'].values[0]) else "N/A"
                f.write(f"  {mk:<15} {vals['IBS']:>10} {vals['IBD']:>10} {vals['Kinship']:>10}\n")
        f.write("\n\n5. OPTIMAL THRESHOLDS (Youden's J)\n" + "-"*80 + "\n")
        td = roc_results[roc_results['Scenario']=='related_vs_unrelated']
        f.write(f"\n  [Related vs Unrelated]\n  {'Marker':<15} {'IBS':>12} {'IBD':>12} {'Kinship':>12}\n  " + "-"*55 + "\n")
        for mk in marker_list:
            vals = {}
            for m in ['IBS','IBD','Kinship']:
                r = td[(td['Marker_Set']==mk) & (td['Metric']==m)]
                vals[m] = f"{r['Optimal_Threshold'].values[0]:.4f}" if len(r)>0 and pd.notna(r['Optimal_Threshold'].values[0]) else "N/A"
            f.write(f"  {mk:<15} {vals['IBS']:>12} {vals['IBD']:>12} {vals['Kinship']:>12}\n")
        f.write("\n\n" + "="*100 + "\nEND OF REPORT\n")
    print(f"  Report: {report_path}")


# ============================================================
# Step 5 Orchestrator
# ============================================================
def step5_evaluate(args, gt_df=None):
    print("\n" + "=" * 70)
    print("STEP 5: Evaluation")
    print("=" * 70)
    outdir = Path(args.outdir)
    results_dir = outdir / "results"
    fig_dir = outdir / "figures"
    reports_dir = outdir / "reports"
    DD = fig_dir/"distributions"; DH = fig_dir/"heatmaps"; DR = fig_dir/"roc_curves"
    DS = fig_dir/"scatter"; DC = fig_dir/"comparison"; DM = fig_dir/"summary"
    for d in [DD, DH, DR, DS, DC, DM, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if gt_df is None:
        gt_path = outdir / "family_relationships.csv"
        if not gt_path.exists():
            print(f"  ERROR: {gt_path} not found. Run step 4 first."); return
        gt_df = pd.read_csv(gt_path)
    marker_list = getattr(args, 'marker_list', [])
    if not marker_list:
        for f in results_dir.glob("*_plink.genome"):
            marker_list.append(f.stem.replace("_plink", ""))
        marker_list = sorted(set(marker_list))
    if not marker_list:
        print("  ERROR: No marker results found."); return
    print(f"  Ground truth: {len(gt_df):,} pairs")
    print(f"  Marker sets: {marker_list}")

    # Load results
    print("\n[2] Loading results...")
    all_res = []
    for ms in marker_list:
        merged = merge_results(gt_df, ms, results_dir)
        print(f"  {ms}: {merged['IBS'].notna().sum():,} pairs with data")
        all_res.append(merged)
    all_df = pd.concat(all_res, ignore_index=True)
    all_df.to_csv(outdir / "all_results_combined.csv", index=False)

    # ROC
    print("\n[3] Calculating ROC metrics...")
    roc_results = calculate_all_roc_scenarios(all_df, marker_list)
    roc_results.to_csv(outdir / "roc_results.csv", index=False)

    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    # [4] Degree boxplots
    print("\n[4] Boxplots by DEGREE...")
    for ms in marker_list:
        plot_boxplot_by_degree_all(all_df, ms, DD/f"boxplot_degree_{ms}.png")

    # [5] Relationship boxplots + violins
    print("\n[5] Boxplots/Violins by RELATIONSHIP...")
    for ms in marker_list:
        plot_boxplot_by_relationship(all_df, ms, DD/f"boxplot_relationship_{ms}.png")
        plot_violin_by_relationship(all_df, ms, DD/f"violin_relationship_{ms}.png")

    # [6] Heatmaps
    print("\n[6] Heatmaps...")
    for ms in marker_list:
        for m in ['IBS','IBD','Kinship']:
            plot_heatmap_standard(all_df, ms, m, DH/f"heatmap_{ms}_{m}.png")

    # [7] Per-family heatmaps (ALL families x ALL markers)
    families = sorted(gt_df['Family1'].unique())
    print(f"\n[7] Per-family heatmaps ({len(families)} families)...")
    for ms in marker_list:
        for fam in families:
            for m in ['IBS','Kinship']:
                plot_heatmap_within_family(all_df, ms, m, fam, DH/f"heatmap_family{fam}_{ms}_{m}.png")

    # [8] ROC curves (6 scenarios)
    print("\n[8] ROC curves...")
    roc_scenarios = [
        ('related_vs_unrelated', lambda d: d['Is_Related']==True, lambda d: d['Is_Related']==False, 'Related vs Unrelated'),
        ('close_vs_unrelated', lambda d: d['Degree'].isin([1,2,3,4]), lambda d: d['Is_Related']==False, '1-4촌 vs Unrelated'),
        ('distant_vs_unrelated', lambda d: d['Degree'].isin([5,6]), lambda d: d['Is_Related']==False, '5-6촌 vs Unrelated'),
        ('4th_vs_unrelated', lambda d: d['Degree']==4, lambda d: d['Is_Related']==False, '4촌 vs Unrelated'),
        ('5th_vs_unrelated', lambda d: d['Degree']==5, lambda d: d['Is_Related']==False, '5촌 vs Unrelated'),
        ('6th_vs_unrelated', lambda d: d['Degree']==6, lambda d: d['Is_Related']==False, '6촌 vs Unrelated'),
    ]
    for name, pos, neg, title in roc_scenarios:
        plot_roc_curves(all_df, name, pos, neg, title, marker_list, DR/f"roc_{name}.png")
        print(f"    Saved: roc_{name}.png")

    # [9] AUC heatmap + Adjacent discrimination
    print("\n[9] Performance comparison...")
    for m in ['IBS','IBD','Kinship']:
        plot_auc_heatmap(roc_results, m, marker_list, DC/f"auc_heatmap_{m}.png")
    plot_adjacent_discrimination(roc_results, marker_list, DC/"adjacent_discrimination.png")
    print(f"    Saved: adjacent_discrimination.png")

    # [10] Scatter
    print("\n[10] Scatter plots...")
    for ms in marker_list:
        plot_scatter_expected_vs_observed(all_df, ms, DS/f"scatter_{ms}.png")

    # [11] NEW: Marker comparison overlay
    print("\n[11] Marker comparison overlay...")
    if len(marker_list) > 1:
        for m in ['IBS','IBD','Kinship']:
            plot_marker_comparison_overlay(all_df, marker_list, m, DC/f"marker_overlay_{m}.png")

    # [12] NEW: Per-degree summary stats
    print("\n[12] Per-degree summary statistics...")
    generate_degree_summary_stats(all_df, marker_list, reports_dir/"degree_summary_stats.csv", DM/"degree_summary.png")

    # [13] NEW: Effect size
    print("\n[13] Effect size (Cohen's d)...")
    plot_effect_size_adjacent(all_df, marker_list, DM/"effect_size_adjacent.png")

    # [14] NEW: Confusion matrices
    print("\n[14] Confusion matrices...")
    plot_confusion_matrices(all_df, roc_results, marker_list, DM/"confusion_matrices.png")

    # [15] Report
    print("\n[15] Generating report...")
    generate_report(all_df, roc_results, marker_list, reports_dir/"kinship_analysis_report.txt")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Related vs Unrelated (AUC)")
    print("=" * 70)
    sdf = roc_results[roc_results['Scenario']=='related_vs_unrelated']
    print(f"\n{'Marker':<15} {'IBS':>10} {'IBD':>10} {'Kinship':>10}")
    print("-" * 50)
    for mk in marker_list:
        md = sdf[sdf['Marker_Set']==mk]
        vals = {}
        for m in ['IBS','IBD','Kinship']:
            r = md[md['Metric']==m]['AUC'].values
            vals[m] = f"{r[0]:.4f}" if len(r)>0 and pd.notna(r[0]) else "N/A"
        print(f"{mk:<15} {vals['IBS']:>10} {vals['IBD']:>10} {vals['Kinship']:>10}")
    print(f"\nAll outputs in: {outdir}")


# ============================================================
# qsub chain helper
# ============================================================
def create_eval_script(args):
    outdir = Path(args.outdir)
    scripts_dir = outdir / "scripts"; logs_dir = outdir / "logs"
    scripts_dir.mkdir(parents=True, exist_ok=True); logs_dir.mkdir(parents=True, exist_ok=True)
    this_script = os.path.abspath(__file__)
    families_str = ','.join(str(f) for f in args.family_list)
    marker_args = []
    if args.config_file:
        marker_args.append(f"--config {args.config_file}")
    else:
        mm = {'--36k': args.bed_36k, '--24k': args.bed_24k, '--20k': args.bed_20k,
              '--12k': args.bed_12k, '--6k': args.bed_6k,
              '--kintelligence': args.bed_kintelligence, '--qiaseq': args.bed_qiaseq}
        for flag, val in mm.items():
            if val: marker_args.append(f"{flag} {val}")
    marker_args_str = " \\\n    ".join(marker_args) if marker_args else ""
    script_content = f"""#!/bin/bash
#$ -N eval_kinship
#$ -o {logs_dir}/eval_kinship.out
#$ -e {logs_dir}/eval_kinship.err
#$ -cwd
#$ -V
#$ -pe smp 2

echo "Steps 4+5: Ground Truth + Evaluation"
echo "Start: $(date)"

python3 {this_script} \\
    --families {families_str} \\
    {marker_args_str} \\
    --joint-vcf {args.joint_vcf} \\
    --ped {args.ped} \\
    --outdir {args.outdir} \\
    --start-from 4 \\
    --run-mode local

echo "End: $(date)"
"""
    script_path = scripts_dir / "eval_kinship.sh"
    with open(script_path, 'w') as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    return script_path


# ============================================================
# MAIN
# ============================================================
def main():
    args = parse_args()
    print("=" * 70)
    print("KINSHIP ANALYSIS UNIFIED PIPELINE (v2)")
    print("=" * 70)
    print(f"  Families: {args.family_list}")
    print(f"  Marker sets: {args.marker_list}")
    print(f"  Joint VCF: {args.joint_vcf}")
    print(f"  PED file: {args.ped}")
    print(f"  Output: {args.outdir}")
    print(f"  Run mode: {args.run_mode}")
    print(f"  Start from: Step {args.start_from}")
    if args.dry_run: print(f"  *** DRY RUN ***")

    if args.start_from <= 3:
        kin_job_names = step3_kinship_analysis(args)
        if args.run_mode == 'qsub' and not args.dry_run and kin_job_names:
            eval_script = create_eval_script(args)
            hold_jid = ','.join(kin_job_names)
            result = subprocess.run(f"qsub -hold_jid {hold_jid} {eval_script}", shell=True, capture_output=True, text=True)
            print(f"\n  Submitted eval job (depends on {hold_jid}): {result.stdout.strip()}")
            print(f"  Monitor: qstat")
            print(f"  Logs: {Path(args.outdir)/'logs'}")
            return
        elif args.run_mode == 'qsub' and args.dry_run:
            eval_script = create_eval_script(args)
            hold_jid = ','.join(kin_job_names) if kin_job_names else 'kin_*'
            print(f"\n  [DRY-RUN] qsub -hold_jid {hold_jid} {eval_script}")
            return

    gt_df = None
    if args.start_from <= 4:
        gt_df = step4_ground_truth(args)
    if args.start_from <= 5:
        step5_evaluate(args, gt_df)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
