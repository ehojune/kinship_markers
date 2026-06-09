#!/usr/bin/env python3
"""
Step 4: Ground truth generation
===============================
Generate pairwise pedigree-derived relationship ground truth from a PED file.
Outputs are written to 06_kinship_analysis/family_relationships.csv by default.
"""

import argparse
from pathlib import Path
from collections import defaultdict
from itertools import combinations

import pandas as pd

HOME = Path.home()
DDEFAULT_WORK_DIR  = "/mnt/d/Research/20251031_wgrs"
DEFAULT_JOINT_VCF = "/mnt/d/Research/20251031_wgrs/05_jointcall/joint_called.allsites.vcf.gz"
DEFAULT_ANALYSIS_DIR = "/mnt/d/Research/20251031_wgrs/06_kinship_analysis"

ALL_FAMILIES = [1, 2, 4, 5, 6, 9, 10, 14, 15, 18]

class FamilyTree:
    """Pedigree relationship inference with corrected kinship coefficients.
    Fixes: Grand-Uncle-Nephew=4, LCA path multiplier in Wright's formula,
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
            rel = names.get(chon, f"Direct-{chon}")
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
            rel = f"Distant-{chon}"
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
    analysis_dir = Path(args.analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    gt_path = analysis_dir / "family_relationships.csv"
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
        print(f"  {deg:>4} {len(sub):>6}  {rel_str}")
    return gt_df

def parse_args():
    parser = argparse.ArgumentParser(description='Step 4: generate pedigree ground truth')
    parser.add_argument('--families', required=True,
                        help='"all" or comma-separated (e.g. "1,2,5,6")')
    parser.add_argument('--ped', default=f"{DEFAULT_ANALYSIS_DIR}/full_pedigree.ped",)
    parser.add_argument('--analysis-dir', '--outdir', dest='analysis_dir', default=str(DEFAULT_ANALYSIS_DIR),
                        help='Directory for Step 4 output (default: 06_kinship_analysis)')
    args = parser.parse_args()
    if args.families.lower() == 'all':
        args.family_list = ALL_FAMILIES
    else:
        args.family_list = [int(f.strip()) for f in args.families.split(',')]
    return args


def main():
    args = parse_args()
    print("=" * 70)
    print("STEP 4: GROUND TRUTH GENERATION")
    print("=" * 70)
    print(f"  Families: {args.family_list}")
    print(f"  PED file: {args.ped}")
    print(f"  Step 4 output: {args.analysis_dir}")
    if step4_ground_truth(args) is None:
        raise SystemExit(1)
    print("\nSTEP 4 COMPLETE")


if __name__ == "__main__":
    main()
