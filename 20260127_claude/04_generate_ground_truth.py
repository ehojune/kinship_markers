#!/usr/bin/env python3
"""
Step 4: Generate Family Relationship Ground Truth
EXCLUDING families 004 and 009

Uses same PED file but filters out specified families
"""

import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from collections import defaultdict

# ============================================================
# Configuration
# ============================================================
HOME = Path.home()
PED_FILE = HOME / "kinship/Analysis/20251031_wgrs/full_pedigree.ped"
OUTPUT_DIR = HOME / "kinship/Analysis/20251031_wgrs/06_kinship_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "family_relationships.csv"

# Families to EXCLUDE
EXCLUDE_FAMILIES = []


class FamilyTree:
    """Class to represent and analyze a family pedigree"""
    
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
        member = self.members[member_id]
        parents = []
        if member['father']:
            parents.append(member['father'])
        if member['mother']:
            parents.append(member['mother'])
        return parents
    
    def get_all_ancestors(self, member_id, max_depth=10):
        ancestors = {}
        def trace_ancestors(current_id, depth):
            if depth > max_depth:
                return
            for parent_id in self.get_parents(current_id):
                if parent_id not in ancestors or ancestors[parent_id] > depth:
                    ancestors[parent_id] = depth
                    trace_ancestors(parent_id, depth + 1)
        trace_ancestors(member_id, 1)
        return ancestors
    
    def find_lowest_common_ancestor(self, id1, id2):
        if id1 == id2:
            return (id1, 0, 0)
        
        ancestors1 = self.get_all_ancestors(id1)
        ancestors1[id1] = 0
        ancestors2 = self.get_all_ancestors(id2)
        ancestors2[id2] = 0
        
        common = set(ancestors1.keys()) & set(ancestors2.keys())
        
        if not common:
            return (None, None, None)
        
        best_lca = None
        best_dist1 = float('inf')
        best_dist2 = float('inf')
        
        for ancestor in common:
            d1 = ancestors1[ancestor]
            d2 = ancestors2[ancestor]
            if d1 + d2 < best_dist1 + best_dist2:
                best_lca = ancestor
                best_dist1 = d1
                best_dist2 = d2
        
        return (best_lca, best_dist1, best_dist2)
    
    def get_relationship_type(self, id1, id2):
        if id1 == id2:
            return ("Self", 0, 0.5)
        
        children1 = set(self.children.get(id1, []))
        children2 = set(self.children.get(id2, []))
        common_children = children1 & children2
        
        lca, dist1, dist2 = self.find_lowest_common_ancestor(id1, id2)
        
        if lca is None:
            if common_children:
                return ("Spouse", 0, 0.0)
            else:
                return ("Unrelated", 0, 0.0)
        
        degree = dist1 + dist2
        
        if dist1 == 0 or dist2 == 0:
            if dist1 == 1 or dist2 == 1:
                return ("Parent-Child", 1, 0.25)
            elif dist1 == 2 or dist2 == 2:
                return ("Grandparent-Grandchild", 2, 0.125)
            elif dist1 == 3 or dist2 == 3:
                return ("Great-Grandparent", 3, 0.0625)
            else:
                return (f"Ancestor-Descendant", degree, (0.5) ** degree)
        
        elif dist1 == 1 and dist2 == 1:
            return ("Sibling", 2, 0.25)
        
        elif (dist1 == 1 and dist2 == 2) or (dist1 == 2 and dist2 == 1):
            return ("Uncle-Nephew", 3, 0.0625)
        
        elif dist1 == 2 and dist2 == 2:
            return ("Cousin", 4, 0.03125)
        
        elif (dist1 == 1 and dist2 == 3) or (dist1 == 3 and dist2 == 1):
            return ("Grand-Uncle-Nephew", 5, 0.015625)
        
        elif (dist1 == 2 and dist2 == 3) or (dist1 == 3 and dist2 == 2):
            return ("Cousin-Once-Removed", 5, 0.015625)
        
        elif dist1 == 3 and dist2 == 3:
            return ("Second-Cousin", 6, 0.0078125)
        
        elif (dist1 == 1 and dist2 == 4) or (dist1 == 4 and dist2 == 1):
            return ("Great-Grand-Uncle", 6, 0.0078125)
        
        elif (dist1 == 2 and dist2 == 4) or (dist1 == 4 and dist2 == 2):
            return ("Cousin-Twice-Removed", 6, 0.0078125)
        
        else:
            expected_kinship = (0.5) ** (degree)
            return (f"Distant-{degree}촌", degree, expected_kinship)


def parse_ped_file(ped_path):
    """Parse PED file and build family trees, excluding specified families"""
    
    families = {}
    sample_to_family = {}
    
    print(f"  Reading: {ped_path}")
    print(f"  Excluding families: {EXCLUDE_FAMILIES}")
    
    excluded_count = 0
    
    with open(ped_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            fields = line.split('\t')
            if len(fields) < 5:
                fields = line.split()
            
            if len(fields) < 5:
                continue
            
            fam_id = fields[0]
            ind_id = fields[1]
            
            # Check if this family should be excluded
            fam_num = fam_id.replace('FAM', '')
            if fam_id in EXCLUDE_FAMILIES or fam_num in EXCLUDE_FAMILIES:
                excluded_count += 1
                continue
            
            father_id = fields[2] if fields[2] and fields[2] != '0' else None
            mother_id = fields[3] if fields[3] and fields[3] != '0' else None
            sex = int(fields[4]) if fields[4].isdigit() else 0
            
            if fam_id not in families:
                families[fam_id] = FamilyTree(fam_id)
            
            families[fam_id].add_member(ind_id, father_id, mother_id, sex)
            sample_to_family[ind_id] = (fam_id, ind_id)
    
    print(f"  Excluded {excluded_count} individuals from families 004/009")
    
    return families, sample_to_family


def generate_ground_truth(families, sample_to_family):
    """Generate ground truth for all sample pairs"""
    
    results = []
    all_samples = list(sample_to_family.keys())
    
    print(f"  Total individuals (after exclusion): {len(all_samples)}")
    
    # Within-family pairs
    print("  Processing within-family pairs...")
    for fam_id, tree in families.items():
        members = list(tree.members.keys())
        
        for id1, id2 in combinations(members, 2):
            rel_type, degree, expected_kin = tree.get_relationship_type(id1, id2)
            
            parts1 = id1.split('-')
            parts2 = id2.split('-')
            
            fam_num = parts1[1] if len(parts1) >= 2 else fam_id
            member1 = parts1[2] if len(parts1) >= 3 else id1
            member2 = parts2[2] if len(parts2) >= 3 else id2
            
            results.append({
                'Sample1': id1,
                'Sample2': id2,
                'Family1': fam_num,
                'Family2': fam_num,
                'Member1': member1,
                'Member2': member2,
                'Relationship': rel_type,
                'Degree': degree if degree else 0,
                'Expected_Kinship': expected_kin,
                'Same_Family': True,
                'Is_Related': degree is not None and degree > 0
            })
    
    # Between-family pairs
    print("  Processing between-family pairs...")
    family_ids = list(families.keys())
    
    for i, fam1 in enumerate(family_ids):
        for fam2 in family_ids[i+1:]:
            members1 = list(families[fam1].members.keys())
            members2 = list(families[fam2].members.keys())
            
            for id1 in members1:
                for id2 in members2:
                    parts1 = id1.split('-')
                    parts2 = id2.split('-')
                    
                    fam_num1 = parts1[1] if len(parts1) >= 2 else fam1
                    fam_num2 = parts2[1] if len(parts2) >= 2 else fam2
                    member1 = parts1[2] if len(parts1) >= 3 else id1
                    member2 = parts2[2] if len(parts2) >= 3 else id2
                    
                    results.append({
                        'Sample1': id1,
                        'Sample2': id2,
                        'Family1': fam_num1,
                        'Family2': fam_num2,
                        'Member1': member1,
                        'Member2': member2,
                        'Relationship': 'Unrelated',
                        'Degree': 0,
                        'Expected_Kinship': 0.0,
                        'Same_Family': False,
                        'Is_Related': False
                    })
    
    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("Ground Truth Generation (EXCLUDING Families 004 and 009)")
    print("=" * 70)
    
    print("\n[1] Parsing PED file...")
    
    if not PED_FILE.exists():
        print(f"  ERROR: PED file not found: {PED_FILE}")
        return
    
    families, sample_to_family = parse_ped_file(PED_FILE)
    
    print(f"  Found {len(families)} families (after exclusion)")
    for fam_id in sorted(families.keys()):
        n_members = len(families[fam_id].members)
        print(f"    {fam_id}: {n_members} members")
    
    print("\n[2] Calculating relationships...")
    gt_df = generate_ground_truth(families, sample_to_family)
    
    gt_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved: {OUTPUT_FILE}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Relationship Summary")
    print("=" * 70)
    
    print(f"\n[All Pairs]")
    print(f"  Total pairs: {len(gt_df):,}")
    
    print("\n[By Relationship Type]")
    rel_counts = gt_df['Relationship'].value_counts()
    for rel, count in rel_counts.items():
        print(f"  {rel:<30}: {count:>5} pairs")
    
    print("\n[By Degree]")
    within_fam = gt_df[gt_df['Same_Family'] == True]
    degree_counts = within_fam.groupby('Degree').size()
    for deg, count in degree_counts.items():
        label = f"{deg}촌" if deg > 0 else "Unrelated (Spouse)"
        print(f"  {label:<25}: {count:>5} pairs")


if __name__ == "__main__":
    main()