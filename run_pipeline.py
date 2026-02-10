#!/usr/bin/env python3
"""
Kinship Analysis Unified Pipeline
===================================
Single command to run the entire post-joint-calling pipeline:
  Step 3: Extract VCF subsets -> PLINK IBS/IBD -> KING kinship (per marker set, qsub parallel)
  Step 4: Generate ground truth from PED file (qsub after Step 3)
  Step 5: Evaluate results and generate ALL figures (qsub after Step 4)

ALL plotting functions from 05_evaluate_results.py are included:
  - Boxplot by degree (with Spouse/InLaw, Between-Fam labels, DEGREE_COLORS)
  - Boxplot by relationship type (with RELATIONSHIP_COLORS)
  - Violin plot by relationship type
  - Full-sample heatmap (all samples, per marker, per metric, RdYlBu_r)
  - Per-family heatmap (ALL families)
  - ROC curves (5 key scenarios)
  - AUC heatmap (marker x all 12 scenarios)
  - Adjacent degree discrimination
  - Scatter expected vs observed
  - Text report with optimal thresholds

Usage examples:
  python run_pipeline.py --families all \
      --36k beds/NFS_36K.bed --24k beds/NFS_24K.bed --20k beds/NFS_20K.bed \
      --12k beds/NFS_12K.bed --6k beds/NFS_6K.bed

  python run_pipeline.py --families 1,2,5,6,10,14,15,18 --36k beds/NFS_36K.bed

  python run_pipeline.py --families all --36k beds/NFS_36K.bed --start-from 4
"""

import argparse
import subprocess
import os
import sys
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
from sklearn.metrics import roc_auc_score, roc_curve
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
# Plotting Constants (from 05_evaluate_results.py)
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
# Argument Parsing
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description='Kinship Analysis Unified Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
        Examples:
          python run_pipeline.py --families all --36k beds/NFS_36K.bed --12k beds/NFS_12K.bed
          python run_pipeline.py --families 1,2,5,6 --36k beds/NFS_36K.bed --start-from 4
        """))

    parser.add_argument('--families', required=True,
                        help='Family numbers: "all" or comma-separated (e.g., "1,2,5,6,10,14,15,18")')

    mg = parser.add_argument_group('Marker sets (provide at least one)')
    mg.add_argument('--36k', dest='bed_36k', help='BED file for NFS_36K')
    mg.add_argument('--24k', dest='bed_24k', help='BED file for NFS_24K')
    mg.add_argument('--20k', dest='bed_20k', help='BED file for NFS_20K')
    mg.add_argument('--12k', dest='bed_12k', help='BED file for NFS_12K')
    mg.add_argument('--6k',  dest='bed_6k',  help='BED file for NFS_6K')
    mg.add_argument('--kintelligence', dest='bed_kintelligence', help='BED file for Kintelligence')
    mg.add_argument('--qiaseq', dest='bed_qiaseq', help='BED file for QIAseq')

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

    args.marker_sets = {}
    mm = {'NFS_36K': args.bed_36k, 'NFS_24K': args.bed_24k, 'NFS_20K': args.bed_20k,
          'NFS_12K': args.bed_12k, 'NFS_6K': args.bed_6k,
          'Kintelligence': args.bed_kintelligence, 'QIAseq': args.bed_qiaseq}
    for name, path in mm.items():
        if path:
            p = Path(path)
            if not p.exists():
                parser.error(f"BED file not found: {path}")
            args.marker_sets[name] = p

    if not args.marker_sets and args.start_from <= 3:
        parser.error("At least one marker set BED file required")

    args.marker_list = [m for m in ['NFS_36K','NFS_24K','NFS_20K','NFS_12K','NFS_6K',
                                     'Kintelligence','QIAseq'] if m in args.marker_sets]
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

    outdir    = Path(args.outdir)
    vcf_dir   = outdir / "vcf_subsets"
    plink_dir = outdir / "plink_files"
    results_dir = outdir / "results"
    scripts_dir = outdir / "scripts"
    logs_dir  = outdir / "logs"

    for d in [vcf_dir, plink_dir, results_dir, scripts_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    joint_vcf = Path(args.joint_vcf)
    samples = get_sample_list(args.family_list)

    sample_file = outdir / "selected_samples.txt"
    with open(sample_file, 'w') as f:
        for s in samples:
            f.write(s + '\n')
    print(f"  Samples: {len(samples)} from families {args.family_list}")
    print(f"  Sample list: {sample_file}")

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

# Step 1: Extract VCF subset
echo ""
echo "[1/4] Extracting VCF subset..."
bcftools view -S {sample_file} -R {bed_path} {joint_vcf} -Oz -o {vcf_out}
tabix -p vcf {vcf_out}

TOTAL=$(bcftools view -H {vcf_out} | wc -l)
SNPS=$(bcftools view -H -v snps {vcf_out} | wc -l)
echo "  Total sites: $TOTAL"
echo "  SNP sites: $SNPS"

# Step 2: VCF -> PLINK binary
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

# Step 3: IBS/IBD (PLINK --genome)
echo ""
echo "[3/4] Calculating IBS/IBD..."
plink --bfile {plink_prefix} \\
      --genome \\
      --out {results_prefix}_plink \\
      --allow-extra-chr \\
      2>&1 | tail -3

echo "  Pairs: $(tail -n +2 {results_prefix}_plink.genome | wc -l)"

# Step 4: Kinship coefficient (KING)
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
                result = subprocess.run(f"qsub {script_path}", shell=True,
                                        capture_output=True, text=True)
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
    def __init__(self, family_id):
        self.family_id = family_id
        self.members = {}
        self.children = defaultdict(list)

    def add_member(self, member_id, father_id, mother_id, sex):
        self.members[member_id] = {
            'father': father_id if father_id else None,
            'mother': mother_id if mother_id else None,
            'sex': sex
        }
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
        """Returns (relationship_name, korean_chon, wright_kinship)"""
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
            fam_id = fields[0]
            ind_id = fields[1]
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
                'Expected_Kinship': ek,
                'Same_Family': True, 'Is_Related': chon > 0
            })

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
                        'Expected_Kinship': 0.0,
                        'Same_Family': False, 'Is_Related': False
                    })

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
# Step 5: Plot 1 - Boxplot by Degree (Spouse/InLaw, Between-Fam)
# ============================================================

def plot_boxplot_by_degree_all(all_df, marker_set, output_path):
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    if len(df) == 0:
        return

    def get_degree_label(row):
        if row['Same_Family']:
            if row['Degree'] == 0:
                return 'Spouse/InLaw\n(0촌)'
            else:
                return f"{row['Degree']}촌"
        else:
            return 'Between-Fam\n(Unrel)'

    df['Degree_Label'] = df.apply(get_degree_label, axis=1)
    order = ['1촌', '2촌', '3촌', '4촌', '5촌', '6촌', 'Spouse/InLaw\n(0촌)', 'Between-Fam\n(Unrel)']
    available_order = [o for o in order if o in df['Degree_Label'].values]

    palette = []
    for o in available_order:
        if '1촌' in o:   palette.append(DEGREE_COLORS[1])
        elif '2촌' in o: palette.append(DEGREE_COLORS[2])
        elif '3촌' in o: palette.append(DEGREE_COLORS[3])
        elif '4촌' in o: palette.append(DEGREE_COLORS[4])
        elif '5촌' in o: palette.append(DEGREE_COLORS[5])
        elif '6촌' in o: palette.append(DEGREE_COLORS[6])
        else:            palette.append(DEGREE_COLORS[0])

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    metrics = ['IBS', 'IBD', 'Kinship']

    for ax, metric in zip(axes, metrics):
        data = df.dropna(subset=[metric])
        if len(data) == 0:
            ax.set_title(f'{metric} (No Data)')
            continue
        sns.boxplot(data=data, x='Degree_Label', y=metric, order=available_order,
                    palette=palette, ax=ax, width=0.6, linewidth=1.5,
                    flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.3})
        sns.stripplot(data=data, x='Degree_Label', y=metric, order=available_order,
                      color='black', size=2, alpha=0.3, ax=ax, jitter=True)
        for i, deg in enumerate(available_order):
            n = len(data[data['Degree_Label'] == deg])
            ymin = data[metric].min() - (data[metric].max() - data[metric].min()) * 0.1
            ax.annotate(f'n={n}', xy=(i, ymin), ha='center', va='top',
                        fontsize=9, color='gray', style='italic')
        ax.set_xlabel('Degree', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'{marker_set} - Distribution by Degree (All Pairs)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {output_path.name}")


# ============================================================
# Step 5: Plot 2 - Boxplot by Relationship (RELATIONSHIP_COLORS)
# ============================================================

def plot_boxplot_by_relationship(all_df, marker_set, output_path):
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    if len(df) == 0:
        return
    rel_order = [r for r in RELATIONSHIP_ORDER if r in df['Relationship'].values]
    if len(rel_order) == 0:
        return
    df['Rel_Label'] = df['Relationship'].map(lambda x: RELATIONSHIP_LABELS.get(x, x))
    label_order = [RELATIONSHIP_LABELS.get(r, r) for r in rel_order]
    palette = [RELATIONSHIP_COLORS.get(r, '#95a5a6') for r in rel_order]

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    for ax, metric in zip(axes, ['IBS', 'IBD', 'Kinship']):
        data = df.dropna(subset=[metric])
        if len(data) == 0:
            continue
        sns.boxplot(data=data, x='Rel_Label', y=metric, order=label_order,
                    palette=palette, ax=ax, width=0.6, linewidth=1.5)
        sns.stripplot(data=data, x='Rel_Label', y=metric, order=label_order,
                      color='black', size=2, alpha=0.3, ax=ax, jitter=True)
        for i, label in enumerate(label_order):
            n = len(data[data['Rel_Label'] == label])
            ymin = data[metric].min() - (data[metric].max() - data[metric].min()) * 0.08
            ax.annotate(f'n={n}', xy=(i, ymin), ha='center', va='top',
                        fontsize=8, color='gray', style='italic')
        ax.set_xlabel('Relationship', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'{marker_set} - Distribution by Relationship Type', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {output_path.name}")


# ============================================================
# Step 5: Plot 3 - Violin by Relationship
# ============================================================

def plot_violin_by_relationship(all_df, marker_set, output_path):
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    if len(df) == 0:
        return
    rel_order = [r for r in RELATIONSHIP_ORDER if r in df['Relationship'].values]
    if len(rel_order) == 0:
        return
    df['Rel_Label'] = df['Relationship'].map(lambda x: RELATIONSHIP_LABELS.get(x, x))
    label_order = [RELATIONSHIP_LABELS.get(r, r) for r in rel_order]
    palette = [RELATIONSHIP_COLORS.get(r, '#95a5a6') for r in rel_order]

    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    for ax, metric in zip(axes, ['IBS', 'IBD', 'Kinship']):
        data = df.dropna(subset=[metric])
        if len(data) == 0:
            continue
        sns.violinplot(data=data, x='Rel_Label', y=metric, order=label_order,
                       palette=palette, ax=ax, inner='box', linewidth=1)
        ax.set_xlabel('Relationship', fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'{marker_set} - Violin Plot by Relationship', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# Step 5: Plot 4 - Full-sample Heatmap (RdYlBu_r)
# ============================================================

def plot_heatmap_standard(all_df, marker_set, metric, output_path):
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    if len(df) == 0 or df[metric].isna().all():
        return

    samples = sorted(set(df['Sample1'].tolist() + df['Sample2'].tolist()))
    samples = [s for s in samples if 'GP' not in s and 'GM' not in s]
    n = len(samples)
    if n == 0:
        return

    matrix = np.full((n, n), np.nan)
    sample_idx = {s: i for i, s in enumerate(samples)}
    for _, row in df.iterrows():
        s1, s2 = str(row['Sample1']), str(row['Sample2'])
        i, j = sample_idx.get(s1), sample_idx.get(s2)
        val = row[metric]
        if i is not None and j is not None and pd.notna(val):
            matrix[i, j] = val
            matrix[j, i] = val

    if metric == 'Kinship':
        np.fill_diagonal(matrix, 0.5)
    else:
        np.fill_diagonal(matrix, 1.0)

    figsize = max(14, n * 0.25)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.9))

    if metric == 'IBS':
        vmin, vmax = 0.55, 0.85
    elif metric == 'IBD':
        vmin, vmax = 0, 0.6
    else:
        vmin, vmax = -0.05, 0.3

    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    sns.heatmap(matrix, mask=mask, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                square=True, linewidths=0.2, linecolor='white',
                cbar_kws={'shrink': 0.6, 'label': metric}, ax=ax)

    sample_labels = []
    for s in samples:
        parts = str(s).split('-')
        if len(parts) >= 3:
            sample_labels.append(f"{parts[1]}-{parts[2]}")
        else:
            sample_labels.append(str(s))

    fontsize = max(4, min(8, 120 // n))
    ax.set_xticks(np.arange(n) + 0.5)
    ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(sample_labels, rotation=90, ha='center', fontsize=fontsize)
    ax.set_yticklabels(sample_labels, rotation=0, ha='right', fontsize=fontsize)
    ax.set_title(f'{marker_set} - {metric}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {output_path.name}")


# ============================================================
# Step 5: Plot 5 - Per-family Heatmap
# ============================================================

def plot_heatmap_within_family(all_df, marker_set, metric, family, output_path):
    df = all_df[(all_df['Marker_Set'] == marker_set) &
                (all_df['Family1'] == family) &
                (all_df['Family2'] == family)].copy()
    if len(df) == 0:
        return

    samples = sorted(set(df['Sample1'].tolist() + df['Sample2'].tolist()))
    samples = [s for s in samples if 'GP' not in s and 'GM' not in s]
    n = len(samples)
    if n == 0:
        return

    matrix = np.full((n, n), np.nan)
    sample_idx = {s: i for i, s in enumerate(samples)}
    for _, row in df.iterrows():
        s1, s2 = str(row['Sample1']), str(row['Sample2'])
        i, j = sample_idx.get(s1), sample_idx.get(s2)
        val = row[metric]
        if i is not None and j is not None and pd.notna(val):
            matrix[i, j] = val
            matrix[j, i] = val

    if metric == 'Kinship':
        np.fill_diagonal(matrix, 0.5)
    else:
        np.fill_diagonal(matrix, 1.0)

    fig, ax = plt.subplots(figsize=(10, 9))
    if metric == 'IBS':
        vmin, vmax = 0.6, 0.85
    elif metric == 'IBD':
        vmin, vmax = 0, 0.55
    else:
        vmin, vmax = -0.05, 0.3

    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    sns.heatmap(matrix, mask=mask, cmap='RdYlBu_r', vmin=vmin, vmax=vmax,
                square=True, linewidths=0.5, linecolor='white',
                annot=True, fmt='.3f', annot_kws={'size': 9},
                cbar_kws={'shrink': 0.7, 'label': metric}, ax=ax)

    sample_labels = [s.split('-')[-1] for s in samples]
    ax.set_xticklabels(sample_labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(sample_labels, rotation=0, fontsize=11)
    ax.set_title(f'Family {family} - {marker_set} - {metric}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# Step 5: ROC calculation (all 12 scenarios + optimal threshold)
# ============================================================

def calculate_roc_metrics(y_true, y_score):
    valid_mask = ~np.isnan(y_score)
    y_true = np.array(y_true)[valid_mask]
    y_score = np.array(y_score)[valid_mask]
    if len(np.unique(y_true)) < 2 or len(y_true) == 0:
        return None, None, None, None
    try:
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        return auc, fpr, tpr, thresholds
    except:
        return None, None, None, None


def calculate_all_roc_scenarios(all_df, marker_list):
    scenarios = {
        'related_vs_unrelated': {
            'pos': lambda d: d['Is_Related'] == True,
            'neg': lambda d: d['Is_Related'] == False,
            'desc': 'Related vs Unrelated (All)'},
        'blood_within_vs_unrelated': {
            'pos': lambda d: (d['Same_Family'] == True) & (d['Degree'] > 0),
            'neg': lambda d: d['Is_Related'] == False,
            'desc': 'Blood Relatives vs Unrelated'},
        'close_vs_unrelated': {
            'pos': lambda d: d['Degree'].isin([1,2,3,4]),
            'neg': lambda d: d['Is_Related'] == False,
            'desc': '1-4촌 vs Unrelated'},
        'distant_vs_unrelated': {
            'pos': lambda d: d['Degree'].isin([5,6]),
            'neg': lambda d: d['Is_Related'] == False,
            'desc': '5-6촌 vs Unrelated'},
        '1st_vs_2nd': {
            'pos': lambda d: d['Degree'] == 1,
            'neg': lambda d: d['Degree'] == 2,
            'desc': '1촌 vs 2촌'},
        '2nd_vs_3rd': {
            'pos': lambda d: d['Degree'] == 2,
            'neg': lambda d: d['Degree'] == 3,
            'desc': '2촌 vs 3촌'},
        '3rd_vs_4th': {
            'pos': lambda d: d['Degree'] == 3,
            'neg': lambda d: d['Degree'] == 4,
            'desc': '3촌 vs 4촌'},
        '4th_vs_5th': {
            'pos': lambda d: d['Degree'] == 4,
            'neg': lambda d: d['Degree'] == 5,
            'desc': '4촌 vs 5촌'},
        '5th_vs_6th': {
            'pos': lambda d: d['Degree'] == 5,
            'neg': lambda d: d['Degree'] == 6,
            'desc': '5촌 vs 6촌'},
        '4th_vs_unrelated': {
            'pos': lambda d: d['Degree'] == 4,
            'neg': lambda d: d['Is_Related'] == False,
            'desc': '4촌 vs Unrelated'},
        '12345_vs_6': {
            'pos': lambda d: d['Degree'].isin([1,2,3,4,5]),
            'neg': lambda d: d['Degree'] == 6,
            'desc': '12345 vs 6'},
        '6th_vs_unrelated': {
            'pos': lambda d: d['Degree'] == 6,
            'neg': lambda d: d['Is_Related'] == False,
            'desc': '6촌 vs Unrelated'},
    }

    results = []
    for marker_set in marker_list:
        df = all_df[all_df['Marker_Set'] == marker_set]
        for scenario_name, scenario_def in scenarios.items():
            pos_mask = scenario_def['pos'](df)
            neg_mask = scenario_def['neg'](df)
            pos_data = df[pos_mask]
            neg_data = df[neg_mask]
            if len(pos_data) == 0 or len(neg_data) == 0:
                continue
            combined = pd.concat([pos_data, neg_data])
            y_true = pos_mask[combined.index].astype(int)
            for metric in ['IBS', 'IBD', 'Kinship']:
                y_score = combined[metric].values
                auc, fpr, tpr, thresholds = calculate_roc_metrics(y_true, y_score)
                optimal_threshold = None
                if auc is not None and thresholds is not None:
                    j_scores = tpr - fpr
                    optimal_idx = np.argmax(j_scores)
                    optimal_threshold = thresholds[optimal_idx]
                results.append({
                    'Marker_Set': marker_set,
                    'Scenario': scenario_name,
                    'Description': scenario_def['desc'],
                    'Metric': metric,
                    'AUC': auc,
                    'Optimal_Threshold': optimal_threshold,
                    'N_Positive': len(pos_data),
                    'N_Negative': len(neg_data)
                })
    return pd.DataFrame(results)


# ============================================================
# Step 5: Plot 6 - ROC curves
# ============================================================

def plot_roc_curves(all_df, scenario_name, pos_filter, neg_filter, title, marker_list, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ['IBS', 'IBD', 'Kinship']):
        for marker_set in marker_list:
            df = all_df[all_df['Marker_Set'] == marker_set]
            pos_mask = pos_filter(df)
            neg_mask = neg_filter(df)
            pos_data = df[pos_mask]
            neg_data = df[neg_mask]
            if len(pos_data) == 0 or len(neg_data) == 0:
                continue
            combined = pd.concat([pos_data, neg_data])
            y_true = pos_mask[combined.index].astype(int)
            y_score = combined[metric].values
            auc, fpr, tpr, _ = calculate_roc_metrics(y_true, y_score)
            if auc is not None:
                ax.plot(fpr, tpr, label=f'{marker_set} (AUC={auc:.3f})',
                        color=MARKER_COLORS.get(marker_set, 'gray'), linewidth=2)
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
    plt.suptitle(f'ROC: {title}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# Step 5: Plot 7 - AUC Heatmap
# ============================================================

def plot_auc_heatmap(roc_results, metric, marker_list, output_path):
    data = roc_results[roc_results['Metric'] == metric].copy()
    if len(data) == 0:
        return
    pivot = data.pivot(index='Marker_Set', columns='Scenario', values='AUC')
    marker_order = [m for m in marker_list if m in pivot.index]
    if not marker_order:
        return
    pivot = pivot.reindex(marker_order)
    # Ensure numeric
    pivot = pivot.apply(pd.to_numeric, errors='coerce')
    if pivot.isna().all().all():
        return
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0.5, vmax=1.0, ax=ax, linewidths=0.5,
                cbar_kws={'label': 'AUC', 'shrink': 0.8},
                annot_kws={'size': 9})
    ax.set_title(f'{metric} - AUC by Scenario', fontsize=14, fontweight='bold')
    ax.set_xlabel('Scenario', fontsize=12)
    ax.set_ylabel('Marker Set', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# Step 5: Plot 8 - Adjacent Degree Discrimination
# ============================================================

def plot_adjacent_discrimination(roc_results, marker_list, output_path):
    adjacent = ['1st_vs_2nd', '2nd_vs_3rd', '3rd_vs_4th', '4th_vs_5th', '5th_vs_6th']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, metric in zip(axes, ['IBS', 'IBD', 'Kinship']):
        metric_data = roc_results[(roc_results['Metric'] == metric) &
                                   (roc_results['Scenario'].isin(adjacent))]
        for marker_set in marker_list:
            marker_data = metric_data[metric_data['Marker_Set'] == marker_set]
            y_values = []
            for scenario in adjacent:
                row = marker_data[marker_data['Scenario'] == scenario]
                if len(row) > 0 and pd.notna(row['AUC'].values[0]):
                    y_values.append(row['AUC'].values[0])
                else:
                    y_values.append(np.nan)
            ax.plot(range(len(adjacent)), y_values, 'o-',
                    color=MARKER_COLORS.get(marker_set, 'gray'),
                    label=marker_set, linewidth=2, markersize=8)
        ax.set_xticks(range(len(adjacent)))
        ax.set_xticklabels(['1 vs 2', '2 vs 3', '3 vs 4', '4 vs 5', '5 vs 6'], fontsize=10)
        ax.set_xlabel('Adjacent Degrees (촌)', fontsize=12)
        ax.set_ylabel('AUC', fontsize=12)
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_ylim(0.4, 1.05)
        ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    plt.suptitle('Adjacent Degree Discrimination', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# Step 5: Plot 9 - Scatter Expected vs Observed
# ============================================================

def plot_scatter_expected_vs_observed(all_df, marker_set, output_path):
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    if len(df) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ['IBS', 'IBD', 'Kinship']):
        data = df.dropna(subset=[metric])
        for deg in sorted(data['Degree'].unique()):
            deg_data = data[data['Degree'] == deg]
            color = DEGREE_COLORS.get(deg, '#95a5a6')
            label = f"{deg}촌" if deg > 0 else "Unrel"
            ax.scatter(deg_data['Expected_Kinship'], deg_data[metric],
                       c=color, label=label, alpha=0.6, s=30,
                       edgecolors='white', linewidth=0.3)
        valid = data[['Expected_Kinship', metric]].dropna()
        if len(valid) > 2:
            corr = valid['Expected_Kinship'].corr(valid[metric])
            ax.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                        fontsize=12, fontweight='bold')
        if metric == 'Kinship':
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
                    max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlabel('Expected Kinship', fontsize=12)
        ax.set_ylabel(f'Observed {metric}', fontsize=12)
        ax.set_title(f'{metric}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8, ncol=2)
        ax.grid(alpha=0.3)
    plt.suptitle(f'{marker_set} - Expected vs Observed', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# Step 5: Report Generation
# ============================================================

def generate_report(all_df, roc_results, marker_list, report_path):
    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("KINSHIP MARKER PERFORMANCE EVALUATION REPORT\n")
        f.write("=" * 100 + "\n\n")

        sample_df = all_df[all_df['Marker_Set'] == marker_list[0]]
        n_pairs = len(sample_df)
        n_related = len(sample_df[sample_df['Is_Related'] == True])
        n_within_fam = len(sample_df[sample_df['Same_Family'] == True])

        f.write("1. DATASET SUMMARY\n" + "-" * 50 + "\n")
        f.write(f"  Total pairs: {n_pairs:,}\n")
        f.write(f"  Blood-related pairs: {n_related:,}\n")
        f.write(f"  Within-family pairs: {n_within_fam:,}\n")
        f.write(f"  Between-family pairs: {n_pairs - n_within_fam:,}\n\n")

        f.write("2. RELATIONSHIP DISTRIBUTION\n" + "-" * 50 + "\n")
        rel_counts = sample_df['Relationship'].value_counts()
        for rel, count in rel_counts.items():
            f.write(f"  {rel:<30}: {count:>6}\n")
        f.write("\n")

        f.write("3. DEGREE DISTRIBUTION\n" + "-" * 50 + "\n")
        deg_counts = sample_df.groupby('Degree').agg({
            'Sample1': 'count', 'Expected_Kinship': 'first'
        })
        for deg, row in deg_counts.iterrows():
            label = f"{deg}촌" if deg > 0 else "Unrelated"
            f.write(f"  {label:<15}: {int(row['Sample1']):>6} pairs  "
                    f"(phi = {row['Expected_Kinship']:.4f})\n")
        f.write("\n")

        f.write("4. CLASSIFICATION PERFORMANCE (AUC)\n" + "-" * 80 + "\n")

        for sname in ['related_vs_unrelated', 'close_vs_unrelated',
                        '12345_vs_6', '6th_vs_unrelated',
                       'distant_vs_unrelated', '6th_vs_unrelated']:
            sdata = roc_results[roc_results['Scenario'] == sname]
            if len(sdata) == 0:
                continue
            desc = sdata['Description'].iloc[0] if 'Description' in sdata.columns else sname
            f.write(f"\n  [{desc}]\n")
            f.write(f"  {'Marker':<15} {'IBS':>10} {'IBD':>10} {'Kinship':>10}\n")
            f.write("  " + "-" * 50 + "\n")
            for marker in marker_list:
                vals = {}
                for metric in ['IBS', 'IBD', 'Kinship']:
                    row = sdata[(sdata['Marker_Set'] == marker) & (sdata['Metric'] == metric)]
                    if len(row) > 0 and pd.notna(row['AUC'].values[0]):
                        vals[metric] = f"{row['AUC'].values[0]:.4f}"
                    else:
                        vals[metric] = "N/A"
                f.write(f"  {marker:<15} {vals['IBS']:>10} {vals['IBD']:>10} "
                        f"{vals['Kinship']:>10}\n")

        f.write("\n\n" + "=" * 100 + "\nEND OF REPORT\n")
    print(f"  Report: {report_path}")


# ============================================================
# Step 5 Main Orchestrator
# ============================================================

def step5_evaluate(args, gt_df=None):
    print("\n" + "=" * 70)
    print("STEP 5: Evaluation")
    print("=" * 70)

    outdir = Path(args.outdir)
    results_dir = outdir / "results"
    fig_dir = outdir / "figures"
    reports_dir = outdir / "reports"
    FIG_DIST_DIR = fig_dir / "distributions"
    FIG_HEATMAP_DIR = fig_dir / "heatmaps"
    FIG_ROC_DIR = fig_dir / "roc_curves"
    FIG_SCATTER_DIR = fig_dir / "scatter"
    FIG_COMPARISON_DIR = fig_dir / "comparison"

    for d in [FIG_DIST_DIR, FIG_HEATMAP_DIR, FIG_ROC_DIR, FIG_SCATTER_DIR,
              FIG_COMPARISON_DIR, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load ground truth
    if gt_df is None:
        gt_path = outdir / "family_relationships.csv"
        if not gt_path.exists():
            print(f"  ERROR: {gt_path} not found. Run step 4 first.")
            return
        gt_df = pd.read_csv(gt_path)

    # Determine marker list
    marker_list = getattr(args, 'marker_list', [])
    if not marker_list:
        for f in results_dir.glob("*_plink.genome"):
            ms = f.stem.replace("_plink", "")
            marker_list.append(ms)
        marker_list = sorted(set(marker_list))
    if not marker_list:
        print("  ERROR: No marker set results found.")
        return

    print(f"  Ground truth: {len(gt_df):,} pairs")
    print(f"  Marker sets: {marker_list}")

    # [2] Load all results
    print("\n[2] Loading results...")
    all_results = []
    for ms in marker_list:
        merged = merge_results(gt_df, ms, results_dir)
        n_valid = merged['IBS'].notna().sum()
        print(f"  {ms}: {n_valid:,} pairs with data")
        all_results.append(merged)
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(outdir / "all_results_combined.csv", index=False)

    # [3] ROC
    print("\n[3] Calculating ROC metrics...")
    roc_results = calculate_all_roc_scenarios(all_df, marker_list)
    roc_results.to_csv(outdir / "roc_results.csv", index=False)

    # ==========================================
    # GENERATE ALL FIGURES
    # ==========================================
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    # [4] Distribution by Degree
    print("\n[4] Distribution plots by DEGREE...")
    for ms in marker_list:
        output = FIG_DIST_DIR / f"boxplot_degree_{ms}.png"
        plot_boxplot_by_degree_all(all_df, ms, output)

    # [5] Distribution by Relationship Type (boxplot + violin)
    print("\n[5] Distribution plots by RELATIONSHIP TYPE...")
    for ms in marker_list:
        output = FIG_DIST_DIR / f"boxplot_relationship_{ms}.png"
        plot_boxplot_by_relationship(all_df, ms, output)
        output = FIG_DIST_DIR / f"violin_relationship_{ms}.png"
        plot_violin_by_relationship(all_df, ms, output)

    # [6] Full-sample Heatmaps (ALL marker sets x ALL metrics)
    print("\n[6] Heatmaps...")
    for ms in marker_list:
        for metric in ['IBS', 'IBD', 'Kinship']:
            output = FIG_HEATMAP_DIR / f"heatmap_{ms}_{metric}.png"
            plot_heatmap_standard(all_df, ms, metric, output)

    # [7] Per-family Heatmaps (ALL families, first marker set)
    first_ms = marker_list[0]
    families = sorted(gt_df['Family1'].unique())
    print(f"\n[7] Per-family heatmaps ({first_ms}, {len(families)} families)...")
    for fam in families:
        for metric in ['IBS', 'Kinship']:
            output = FIG_HEATMAP_DIR / f"heatmap_family{fam}_{metric}.png"
            plot_heatmap_within_family(all_df, first_ms, metric, fam, output)

    # [8] ROC curves
    print("\n[8] ROC curves...")
    roc_scenarios = [
        ('related_vs_unrelated',
         lambda d: d['Is_Related'] == True,
         lambda d: d['Is_Related'] == False,
         'Related vs Unrelated'),
        ('close_vs_unrelated',
         lambda d: d['Degree'].isin([1,2,3,4]),
         lambda d: d['Is_Related'] == False,
         '1-4촌 vs Unrelated'),
        ('distant_vs_unrelated',
         lambda d: d['Degree'].isin([5,6]),
         lambda d: d['Is_Related'] == False,
         '5-6촌 vs Unrelated'),
        ('4th_vs_unrelated',
         lambda d: d['Degree'] == 4,
         lambda d: d['Is_Related'] == False,
         '4촌 vs Unrelated'),
        ('6th_vs_unrelated',
         lambda d: d['Degree'] == 6,
         lambda d: d['Is_Related'] == False,
         '6촌 vs Unrelated'),
    ]
    for name, pos, neg, title in roc_scenarios:
        output = FIG_ROC_DIR / f"roc_{name}.png"
        plot_roc_curves(all_df, name, pos, neg, title, marker_list, output)
        print(f"    Saved: roc_{name}.png")

    # [9] Performance comparison charts
    print("\n[9] Performance comparison charts...")
    for metric in ['IBS', 'IBD', 'Kinship']:
        output = FIG_COMPARISON_DIR / f"auc_heatmap_{metric}.png"
        plot_auc_heatmap(roc_results, metric, marker_list, output)
    output = FIG_COMPARISON_DIR / "adjacent_discrimination.png"
    plot_adjacent_discrimination(roc_results, marker_list, output)
    print(f"    Saved: adjacent_discrimination.png")

    # [10] Scatter plots
    print("\n[10] Scatter plots...")
    for ms in marker_list[:3]:
        output = FIG_SCATTER_DIR / f"scatter_{ms}.png"
        plot_scatter_expected_vs_observed(all_df, ms, output)

    # [11] Report
    print("\n[11] Generating report...")
    report_path = reports_dir / "kinship_analysis_report.txt"
    generate_report(all_df, roc_results, marker_list, report_path)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Related vs Unrelated (AUC)")
    print("=" * 70)
    scenario_df = roc_results[roc_results['Scenario'] == 'related_vs_unrelated']
    print(f"\n{'Marker':<15} {'IBS':>10} {'IBD':>10} {'Kinship':>10}")
    print("-" * 50)
    for marker in marker_list:
        marker_data = scenario_df[scenario_df['Marker_Set'] == marker]
        ibs = marker_data[marker_data['Metric'] == 'IBS']['AUC'].values
        ibd = marker_data[marker_data['Metric'] == 'IBD']['AUC'].values
        kin = marker_data[marker_data['Metric'] == 'Kinship']['AUC'].values
        ibs_str = f"{ibs[0]:.4f}" if len(ibs) > 0 and pd.notna(ibs[0]) else "N/A"
        ibd_str = f"{ibd[0]:.4f}" if len(ibd) > 0 and pd.notna(ibd[0]) else "N/A"
        kin_str = f"{kin[0]:.4f}" if len(kin) > 0 and pd.notna(kin[0]) else "N/A"
        print(f"{marker:<15} {ibs_str:>10} {ibd_str:>10} {kin_str:>10}")
    print(f"\nAll outputs in: {outdir}")


# ============================================================
# qsub chain helper
# ============================================================
def create_eval_script(args):
    outdir = Path(args.outdir)
    scripts_dir = outdir / "scripts"
    logs_dir = outdir / "logs"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    this_script = os.path.abspath(__file__)
    families_str = ','.join(str(f) for f in args.family_list)

    marker_args = []
    mm = {'--36k': args.bed_36k, '--24k': args.bed_24k, '--20k': args.bed_20k,
          '--12k': args.bed_12k, '--6k': args.bed_6k,
          '--kintelligence': args.bed_kintelligence, '--qiaseq': args.bed_qiaseq}
    for flag, val in mm.items():
        if val:
            marker_args.append(f"{flag} {val}")
    marker_args_str = " \\\n    ".join(marker_args) if marker_args else ""

    script_content = f"""#!/bin/bash
#$ -N eval_kinship
#$ -o {logs_dir}/eval_kinship.out
#$ -e {logs_dir}/eval_kinship.err
#$ -cwd
#$ -V
#$ -pe smp 2

echo "========================================"
echo "Steps 4+5: Ground Truth + Evaluation"
echo "========================================"
echo "Start: $(date)"

python3 {this_script} \\
    --families {families_str} \\
    {marker_args_str} \\
    --joint-vcf {args.joint_vcf} \\
    --ped {args.ped} \\
    --outdir {args.outdir} \\
    --start-from 4 \\
    --run-mode local

echo ""
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
    print("KINSHIP ANALYSIS UNIFIED PIPELINE")
    print("=" * 70)
    print(f"  Families: {args.family_list}")
    print(f"  Marker sets: {args.marker_list}")
    print(f"  Joint VCF: {args.joint_vcf}")
    print(f"  PED file: {args.ped}")
    print(f"  Output: {args.outdir}")
    print(f"  Run mode: {args.run_mode}")
    print(f"  Start from: Step {args.start_from}")
    if args.dry_run:
        print(f"  *** DRY RUN ***")

    # Step 3
    if args.start_from <= 3:
        kin_job_names = step3_kinship_analysis(args)

        if args.run_mode == 'qsub' and kin_job_names and not args.dry_run:
            eval_script = create_eval_script(args)
            hold_jid = ','.join(kin_job_names)
            cmd = f"qsub -hold_jid {hold_jid} {eval_script}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            print(f"\n  Submitted eval job (depends on {hold_jid}): {result.stdout.strip()}")
            print("\n  Pipeline submitted! All steps will run automatically.")
            print(f"  Monitor: qstat")
            print(f"  Logs: {Path(args.outdir) / 'logs'}")
            return
        elif args.run_mode == 'qsub' and args.dry_run:
            eval_script = create_eval_script(args)
            hold_jid = ','.join(kin_job_names) if kin_job_names else 'kin_*'
            print(f"\n  [DRY-RUN] qsub -hold_jid {hold_jid} {eval_script}")
            return

    # Step 4
    gt_df = None
    if args.start_from <= 4:
        gt_df = step4_ground_truth(args)

    # Step 5
    if args.start_from <= 5:
        step5_evaluate(args, gt_df)

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()