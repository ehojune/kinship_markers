#!/usr/bin/env python3
"""
Step 4: Generate Family Relationship Ground Truth (FIXED VERSION)

Correctly calculates:
- Kinship degree (촌수) using LCA (Lowest Common Ancestor)
- Detailed relationship types (Parent, Sibling, Grandparent, Uncle, Cousin, etc.)

PED file format: FamilyID, IndividualID, FatherID, MotherID, Sex, Phenotype
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
PED_FILE = HOME / "kinship/Analysis/20251031_wgrs/full_pedigree.without4and9.ped"
OUTPUT_DIR = HOME / "kinship/Analysis/20251031_wgrs/06_kinship_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "family_relationships.csv"


class FamilyTree:
    """Class to represent and analyze a family pedigree"""
    
    def __init__(self, family_id):
        self.family_id = family_id
        self.members = {}  # member_id -> {'father': id, 'mother': id, 'sex': int}
        self.children = defaultdict(list)  # parent_id -> [child_ids]
    
    def add_member(self, member_id, father_id, mother_id, sex):
        """Add a member to the family tree"""
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
        """Get parents of a member"""
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
        """Get all ancestors with their generation distance"""
        ancestors = {}  # ancestor_id -> generation_distance
        
        def trace_ancestors(current_id, depth):
            if depth > max_depth:
                return
            for parent_id in self.get_parents(current_id):
                if parent_id not in ancestors or ancestors[parent_id] > depth:
                    ancestors[parent_id] = depth
                    trace_ancestors(parent_id, depth + 1)
        
        trace_ancestors(member_id, 1)
        return ancestors
    
    def get_path_to_ancestor(self, member_id, ancestor_id, max_depth=10):
        """Get the path from member to ancestor, return distance or None"""
        if member_id == ancestor_id:
            return 0
        
        visited = {member_id: 0}
        queue = [(member_id, 0)]
        
        while queue:
            current, depth = queue.pop(0)
            if depth >= max_depth:
                continue
            
            for parent_id in self.get_parents(current):
                if parent_id == ancestor_id:
                    return depth + 1
                if parent_id not in visited:
                    visited[parent_id] = depth + 1
                    queue.append((parent_id, depth + 1))
        
        return None
    
    def find_lowest_common_ancestor(self, id1, id2):
        """
        Find LCA and return (lca_id, dist1, dist2)
        dist1 = generations from id1 to LCA
        dist2 = generations from id2 to LCA
        """
        if id1 == id2:
            return (id1, 0, 0)
        
        # Get all ancestors of id1
        ancestors1 = self.get_all_ancestors(id1)
        ancestors1[id1] = 0  # Include self
        
        # Get all ancestors of id2
        ancestors2 = self.get_all_ancestors(id2)
        ancestors2[id2] = 0  # Include self
        
        # Find common ancestors
        common = set(ancestors1.keys()) & set(ancestors2.keys())
        
        if not common:
            return (None, None, None)
        
        # Find LCA with minimum total distance
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
    
    def calculate_kinship_degree(self, id1, id2):
        """
        Calculate kinship degree (촌수) between two individuals
        
        Korean kinship degree calculation:
        - Parent-Child: 1촌
        - Siblings: 2촌 (1 up + 1 down)
        - Grandparent-Grandchild: 2촌
        - Uncle-Nephew: 3촌 (2 up + 1 down or 1 up + 2 down)
        - Cousin: 4촌 (2 up + 2 down)
        
        Formula: degree = dist_to_LCA_1 + dist_to_LCA_2
        """
        if id1 == id2:
            return 0
        
        # Check if one is parent of the other
        if id2 in self.get_parents(id1):
            return 1
        if id1 in self.get_parents(id2):
            return 1
        
        # Check if siblings (same parents)
        parents1 = set(self.get_parents(id1))
        parents2 = set(self.get_parents(id2))
        if parents1 and parents2 and parents1 & parents2:
            return 2  # Siblings
        
        # Find LCA for other relationships
        lca, dist1, dist2 = self.find_lowest_common_ancestor(id1, id2)
        
        if lca is None:
            return None  # No blood relationship
        
        return dist1 + dist2
    
    def get_relationship_type(self, id1, id2):
        """
        Determine the specific relationship type between two individuals
        Returns (relationship_name, degree, expected_kinship)
        """
        if id1 == id2:
            return ("Self", 0, 0.5)
        
        # Check spouse (share a child but not blood related)
        # In our pedigree, spouses are identified by having a common child
        children1 = set(self.children.get(id1, []))
        children2 = set(self.children.get(id2, []))
        common_children = children1 & children2
        
        # Check blood relationship first
        lca, dist1, dist2 = self.find_lowest_common_ancestor(id1, id2)
        
        if lca is None:
            # No blood relationship
            if common_children:
                return ("Spouse", 0, 0.0)
            else:
                # Check if in-law
                return ("Unrelated", 0, 0.0)
        
        degree = dist1 + dist2
        
        # Determine specific relationship type based on distances
        if dist1 == 0 or dist2 == 0:
            # One is ancestor of the other
            if dist1 == 1 or dist2 == 1:
                return ("Parent-Child", 1, 0.25)
            elif dist1 == 2 or dist2 == 2:
                return ("Grandparent-Grandchild", 2, 0.125)
            elif dist1 == 3 or dist2 == 3:
                return ("Great-Grandparent", 3, 0.0625)
            else:
                return (f"Ancestor-Descendant", degree, (0.5) ** degree)
        
        elif dist1 == 1 and dist2 == 1:
            # Both one step from LCA = Siblings
            return ("Sibling", 2, 0.25)
        
        elif (dist1 == 1 and dist2 == 2) or (dist1 == 2 and dist2 == 1):
            # Uncle/Aunt - Nephew/Niece
            return ("Uncle-Nephew", 3, 0.0625)
        
        elif dist1 == 2 and dist2 == 2:
            # First cousins (4촌)
            return ("Cousin", 4, 0.03125)
        
        elif (dist1 == 1 and dist2 == 3) or (dist1 == 3 and dist2 == 1):
            # Grand Uncle - Grand Nephew (5촌)
            return ("Grand-Uncle-Nephew", 5, 0.015625)
        
        elif (dist1 == 2 and dist2 == 3) or (dist1 == 3 and dist2 == 2):
            # First cousin once removed (5촌)
            return ("Cousin-Once-Removed", 5, 0.015625)
        
        elif dist1 == 3 and dist2 == 3:
            # Second cousins (6촌)
            return ("Second-Cousin", 6, 0.0078125)
        
        elif (dist1 == 1 and dist2 == 4) or (dist1 == 4 and dist2 == 1):
            # Great Grand Uncle (6촌)
            return ("Great-Grand-Uncle", 6, 0.0078125)
        
        elif (dist1 == 2 and dist2 == 4) or (dist1 == 4 and dist2 == 2):
            # First cousin twice removed (6촌)
            return ("Cousin-Twice-Removed", 6, 0.0078125)
        
        elif (dist1 == 3 and dist2 == 4) or (dist1 == 4 and dist2 == 3):
            # Second cousin once removed (7촌)
            return ("Second-Cousin-Once-Removed", 7, 0.00390625)
        
        elif dist1 == 4 and dist2 == 4:
            # Third cousins (8촌)
            return ("Third-Cousin", 8, 0.001953125)
        
        else:
            # Generic distant relative
            expected_kinship = (0.5) ** (degree)
            return (f"Distant-{degree}촌", degree, expected_kinship)


def parse_ped_file(ped_path):
    """Parse PED file and build family trees"""
    
    families = {}  # family_id -> FamilyTree
    sample_to_family = {}  # sample_name -> (family_id, member_id)
    
    print(f"  Reading: {ped_path}")
    
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
            father_id = fields[2] if fields[2] and fields[2] != '0' else None
            mother_id = fields[3] if fields[3] and fields[3] != '0' else None
            sex = int(fields[4]) if fields[4].isdigit() else 0
            
            # Create family tree if not exists
            if fam_id not in families:
                families[fam_id] = FamilyTree(fam_id)
            
            # Extract member code from individual ID (e.g., "2024-001-1A" -> "1A")
            parts = ind_id.split('-')
            if len(parts) >= 3:
                member_code = parts[2]
            else:
                member_code = ind_id
            
            # Add to family tree
            families[fam_id].add_member(ind_id, father_id, mother_id, sex)
            
            # Map sample name to family
            # Sample name could be "2024-001-1A" or just the individual ID
            sample_name = ind_id
            sample_to_family[sample_name] = (fam_id, ind_id)
    
    return families, sample_to_family


def generate_ground_truth(families, sample_to_family):
    """Generate ground truth for all sample pairs"""
    
    results = []
    
    # Get list of all actual samples (exclude founders without data)
    # In your case, GPa, GMa etc. might be founders without actual sequencing data
    # We'll include all for now and filter later if needed
    
    all_samples = list(sample_to_family.keys())
    
    # Filter to only samples that likely have data (exclude GP* if needed)
    # For now, keep all samples
    
    print(f"  Total individuals in PED: {len(all_samples)}")
    
    # Within-family pairs
    print("  Processing within-family pairs...")
    for fam_id, tree in families.items():
        members = list(tree.members.keys())
        
        for id1, id2 in combinations(members, 2):
            # Get relationship
            rel_type, degree, expected_kin = tree.get_relationship_type(id1, id2)
            
            # Extract family number and member codes
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
    print("Step 4: Generating Family Relationship Ground Truth (FIXED)")
    print("=" * 70)
    
    # Parse PED file
    print("\n[1] Parsing PED file...")
    
    if not PED_FILE.exists():
        print(f"  ERROR: PED file not found: {PED_FILE}")
        return
    
    families, sample_to_family = parse_ped_file(PED_FILE)
    
    print(f"  Found {len(families)} families")
    
    # Show family structure
    print("\n  Family structure:")
    for fam_id, tree in sorted(families.items()):
        members = list(tree.members.keys())
        print(f"    {fam_id}: {len(members)} members")
        
        # Show a few example relationships
        if len(members) >= 2:
            id1, id2 = members[0], members[1]
            rel, deg, _ = tree.get_relationship_type(id1, id2)
            m1 = id1.split('-')[-1]
            m2 = id2.split('-')[-1]
    
    # Generate ground truth
    print("\n[2] Calculating relationships...")
    
    gt_df = generate_ground_truth(families, sample_to_family)
    
    # Save
    gt_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved: {OUTPUT_FILE}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Relationship Summary")
    print("=" * 70)
    
    print("\n[All Pairs]")
    print(f"  Total pairs: {len(gt_df):,}")
    
    print("\n[By Relationship Type]")
    rel_counts = gt_df['Relationship'].value_counts()
    for rel, count in rel_counts.items():
        print(f"  {rel:<30}: {count:>5} pairs")
    
    print("\n[By Degree (촌수)]")
    within_fam = gt_df[gt_df['Same_Family'] == True]
    degree_counts = within_fam.groupby('Degree').agg({
        'Sample1': 'count',
        'Expected_Kinship': 'first'
    }).rename(columns={'Sample1': 'Count'})
    
    degree_labels = {0: 'Unrelated (Spouse)', 1: '1촌', 2: '2촌', 3: '3촌', 
                    4: '4촌', 5: '5촌', 6: '6촌', 7: '7촌', 8: '8촌'}
    
    for deg, row in degree_counts.iterrows():
        label = degree_labels.get(deg, f'{deg}촌')
        print(f"  {label:<25}: {int(row['Count']):>5} pairs  (Expected φ = {row['Expected_Kinship']:.4f})")
    
    # Verify specific relationship
    print("\n[Verification: 5c-6D relationships]")
    check = gt_df[
        ((gt_df['Member1'] == '5c') & (gt_df['Member2'] == '6D')) |
        ((gt_df['Member1'] == '6D') & (gt_df['Member2'] == '5c'))
    ]
    if len(check) > 0:
        for _, row in check.iterrows():
            print(f"  {row['Sample1']} - {row['Sample2']}: {row['Relationship']} ({row['Degree']}촌)")
    
    print("\n[Sample Examples - Family 001]")
    fam001 = gt_df[(gt_df['Family1'] == '001') & (gt_df['Family2'] == '001')].head(15)
    for _, row in fam001.iterrows():
        print(f"  {row['Member1']:<5} - {row['Member2']:<5}: {row['Relationship']:<30} ({row['Degree']}촌)")


if __name__ == "__main__":
    main()