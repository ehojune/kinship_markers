#!/usr/bin/env python3
"""
Marker coverage analysis per sample
- Individual GVCFs vs marker sets
- Joint VCF (allsites) per sample vs marker sets
- Shows how well each marker set is covered (variant or ref call)

Updated paths for current pipeline
"""

import gzip
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# Configuration
# ============================================================
HOME = Path.home()
WORK_DIR = HOME / "kinship/Analysis/20251031_wgrs"

# Input paths
GVCF_DIR = WORK_DIR / "04_gvcf"
JOINT_VCF = WORK_DIR / "05_jointcall/joint_called.allsites.vcf.gz"
BED_DIR = WORK_DIR / "06_kinship_analysis/marker_beds"

# Output
OUT_DIR = WORK_DIR / "06_kinship_analysis/coverage_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Marker sets (using generated BED files)
MARKER_SETS = {
    "NFS_36K": BED_DIR / "NFS_36K.bed",
    "NFS_24K": BED_DIR / "NFS_24K.bed",
    "NFS_20K": BED_DIR / "NFS_20K.bed",
    "NFS_12K": BED_DIR / "NFS_12K.bed",
    "NFS_6K": BED_DIR / "NFS_6K.bed",
    "Kintelligence": BED_DIR / "Kintelligence.bed",
    "QIAseq": BED_DIR / "QIAseq.bed",
}

# Also check merged markers
MERGED_BED = Path("/BiO/Access/ehojune/kinship/merged_markers.bed")
if MERGED_BED.exists():
    MARKER_SETS["Merged_All"] = MERGED_BED

MAX_THREADS = 30


def parse_bed_file(filepath):
    """Parse BED file and return set of (chrom, pos) tuples (0-based start)"""
    positions = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            fields = line.split('\t')
            if len(fields) < 2:
                fields = line.split()
            if len(fields) >= 2:
                chrom = fields[0]
                try:
                    pos = int(fields[1])  # BED is 0-based
                    positions.add((chrom, pos))
                except ValueError:
                    continue
    return positions


def analyze_gvcf(args):
    """Analyze single GVCF for marker coverage"""
    gvcf_path, marker_sets_positions = args
    sample_name = gvcf_path.stem.replace('.g.vcf', '').replace('.g', '')
    
    results = {
        'sample': sample_name,
        'total_records': 0,
    }
    
    # Initialize counters for each marker set
    for marker_name in marker_sets_positions.keys():
        results[f'{marker_name}_covered'] = 0
        results[f'{marker_name}_variant'] = 0
    
    # Parse GVCF
    gvcf_positions = set()
    variant_positions = set()
    
    try:
        open_func = gzip.open if str(gvcf_path).endswith('.gz') else open
        mode = 'rt' if str(gvcf_path).endswith('.gz') else 'r'
        
        with open_func(gvcf_path, mode) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                fields = line.split('\t', 8)
                if len(fields) >= 5:
                    chrom = fields[0]
                    try:
                        pos = int(fields[1]) - 1  # Convert VCF 1-based to 0-based
                        alt = fields[4]
                        
                        gvcf_positions.add((chrom, pos))
                        results['total_records'] += 1
                        
                        # Check if it's a variant (not ref or <NON_REF> only)
                        if alt not in ['.', '<NON_REF>', '*']:
                            variant_positions.add((chrom, pos))
                    except ValueError:
                        continue
    except Exception as e:
        print(f"  Error reading {gvcf_path.name}: {e}")
        return results
    
    # Calculate overlap with each marker set
    for marker_name, marker_pos in marker_sets_positions.items():
        covered = len(gvcf_positions & marker_pos)
        variants = len(variant_positions & marker_pos)
        results[f'{marker_name}_covered'] = covered
        results[f'{marker_name}_variant'] = variants
    
    return results


def analyze_joint_vcf_per_sample(vcf_path, marker_sets_positions):
    """Analyze joint VCF for per-sample marker coverage"""
    
    print("  Parsing joint VCF header for sample names...")
    
    sample_names = []
    sample_results = {}
    
    # Pre-create sets for marker positions for faster lookup
    marker_pos_sets = {name: pos_set for name, pos_set in marker_sets_positions.items()}
    
    line_count = 0
    
    try:
        open_func = gzip.open if str(vcf_path).endswith('.gz') else open
        mode = 'rt' if str(vcf_path).endswith('.gz') else 'r'
        
        with open_func(vcf_path, mode) as f:
            for line in f:
                if line.startswith('##'):
                    continue
                    
                if line.startswith('#CHROM'):
                    # Parse header to get sample names
                    fields = line.strip().split('\t')
                    sample_names = fields[9:]  # Samples start from column 10
                    print(f"  Found {len(sample_names)} samples in joint VCF")
                    
                    # Initialize results for each sample
                    for sample in sample_names:
                        sample_results[sample] = {
                            'total_called': 0,
                            'total_variant': 0,
                        }
                        for marker_name in marker_sets_positions.keys():
                            sample_results[sample][f'{marker_name}_covered'] = 0
                            sample_results[sample][f'{marker_name}_variant'] = 0
                    continue
                
                # Parse variant line
                fields = line.strip().split('\t')
                if len(fields) < 10:
                    continue
                
                chrom = fields[0]
                try:
                    pos = int(fields[1]) - 1  # Convert to 0-based
                except ValueError:
                    continue
                
                line_count += 1
                if line_count % 10000 == 0:
                    print(f"    Processed {line_count:,} positions...")
                
                # Check which marker sets this position belongs to
                pos_tuple = (chrom, pos)
                in_marker_sets = {
                    marker_name: pos_tuple in marker_pos 
                    for marker_name, marker_pos in marker_pos_sets.items()
                }
                
                # Skip if not in any marker set (optimization)
                if not any(in_marker_sets.values()):
                    continue
                
                # Parse each sample's genotype
                format_fields = fields[8].split(':')
                gt_idx = format_fields.index('GT') if 'GT' in format_fields else 0
                
                for i, sample in enumerate(sample_names):
                    sample_data = fields[9 + i].split(':')
                    if len(sample_data) > gt_idx:
                        gt = sample_data[gt_idx]
                        
                        # Check if called (not ./.)
                        if gt not in ['./.', '.|.', '.']:
                            sample_results[sample]['total_called'] += 1
                            
                            for marker_name, is_in_set in in_marker_sets.items():
                                if is_in_set:
                                    sample_results[sample][f'{marker_name}_covered'] += 1
                            
                            # Check if variant (has alt allele)
                            if '1' in gt or '2' in gt or '3' in gt:
                                sample_results[sample]['total_variant'] += 1
                                
                                for marker_name, is_in_set in in_marker_sets.items():
                                    if is_in_set:
                                        sample_results[sample][f'{marker_name}_variant'] += 1
        
        print(f"  Processed {line_count:,} total positions")
        
    except Exception as e:
        print(f"  Error reading joint VCF: {e}")
        return {}, []
    
    return sample_results, sample_names


def sample_sort_key(sample_name):
    """Sort key for sample names: 2024-001-1A -> (1, 1, 'a')"""
    import re
    parts = sample_name.split('-')
    if len(parts) >= 3:
        try:
            family_num = int(parts[1])
            member = parts[2]
            match = re.match(r'(\d+)([a-zA-Z]*)', member)
            if match:
                num = int(match.group(1))
                letter = match.group(2).lower()
                return (family_num, num, letter)
        except:
            pass
    return (999, 999, sample_name)


def main():
    print("=" * 100)
    print("Marker Coverage Analysis")
    print("=" * 100)
    
    # Load marker sets
    print("\n[1] Loading marker sets...")
    marker_sets = {}
    for name, path in MARKER_SETS.items():
        if path.exists():
            marker_sets[name] = parse_bed_file(path)
            print(f"  {name:15}: {len(marker_sets[name]):>10,} markers")
        else:
            print(f"  {name:15}: NOT FOUND - {path}")
    
    if not marker_sets:
        print("\n  ERROR: No marker sets found!")
        print("  Please run 00_generate_marker_beds.py first!")
        return
    
    # ========================================
    # Part 1: Individual GVCF analysis
    # ========================================
    print(f"\n[2] Analyzing individual GVCFs...")
    
    gvcf_files = sorted(GVCF_DIR.glob("*.g.vcf.gz"))
    if not gvcf_files:
        gvcf_files = sorted(GVCF_DIR.glob("*.g.vcf"))
    
    gvcf_results = []
    
    if gvcf_files:
        print(f"  Found {len(gvcf_files)} GVCF files")
        print(f"  Running parallel analysis ({MAX_THREADS} threads)...")
        
        tasks = [(gvcf, marker_sets) for gvcf in gvcf_files]
        
        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            futures = {executor.submit(analyze_gvcf, task): task[0].stem for task in tasks}
            completed = 0
            for future in as_completed(futures):
                result = future.result()
                gvcf_results.append(result)
                completed += 1
                if completed % 20 == 0 or completed == len(futures):
                    print(f"    Progress: {completed}/{len(futures)}")
        
        # Sort results by sample name
        gvcf_results.sort(key=lambda x: sample_sort_key(x['sample']))
    else:
        print("  No GVCF files found in", GVCF_DIR)
    
    # ========================================
    # Part 2: Joint VCF per-sample analysis
    # ========================================
    print(f"\n[3] Analyzing joint VCF per sample...")
    
    joint_results = {}
    sample_names = []
    
    if JOINT_VCF.exists():
        joint_results, sample_names = analyze_joint_vcf_per_sample(JOINT_VCF, marker_sets)
    else:
        print(f"  Joint VCF not found: {JOINT_VCF}")
    
    # ========================================
    # Part 3: Generate reports
    # ========================================
    print(f"\n[4] Generating reports...")
    
    # Report 1: Individual GVCF coverage
    if gvcf_results:
        report1 = OUT_DIR / "individual_gvcf_marker_coverage.txt"
        with open(report1, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("Individual GVCF Marker Coverage Report\n")
            f.write("=" * 120 + "\n\n")
            
            for marker_name in sorted(marker_sets.keys()):
                marker_total = len(marker_sets[marker_name])
                f.write(f"\n{'='*80}\n")
                f.write(f"Marker Set: {marker_name} ({marker_total:,} markers)\n")
                f.write(f"{'='*80}\n")
                f.write(f"{'Sample':<20} {'Covered':>12} {'Cov%':>8} {'Variants':>12} {'Var%':>8}\n")
                f.write("-" * 65 + "\n")
                
                for res in gvcf_results:
                    covered = res.get(f'{marker_name}_covered', 0)
                    variant = res.get(f'{marker_name}_variant', 0)
                    cov_pct = covered / marker_total * 100 if marker_total > 0 else 0
                    var_pct = variant / marker_total * 100 if marker_total > 0 else 0
                    f.write(f"{res['sample']:<20} {covered:>12,} {cov_pct:>7.2f}% {variant:>12,} {var_pct:>7.2f}%\n")
                
                # Summary stats
                if gvcf_results:
                    avg_cov = sum(r.get(f'{marker_name}_covered', 0) for r in gvcf_results) / len(gvcf_results)
                    avg_var = sum(r.get(f'{marker_name}_variant', 0) for r in gvcf_results) / len(gvcf_results)
                    f.write("-" * 65 + "\n")
                    f.write(f"{'AVERAGE':<20} {avg_cov:>12,.1f} {avg_cov/marker_total*100:>7.2f}% {avg_var:>12,.1f} {avg_var/marker_total*100:>7.2f}%\n")
        
        print(f"  Written: {report1}")
    
    # Report 2: Joint VCF per-sample coverage
    if joint_results:
        report2 = OUT_DIR / "joint_vcf_per_sample_marker_coverage.txt"
        sorted_samples = sorted(sample_names, key=sample_sort_key)
        
        with open(report2, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("Joint VCF Per-Sample Marker Coverage Report\n")
            f.write("=" * 120 + "\n\n")
            
            for marker_name in sorted(marker_sets.keys()):
                marker_total = len(marker_sets[marker_name])
                f.write(f"\n{'='*80}\n")
                f.write(f"Marker Set: {marker_name} ({marker_total:,} markers)\n")
                f.write(f"{'='*80}\n")
                f.write(f"{'Sample':<20} {'Covered':>12} {'Cov%':>8} {'Variants':>12} {'Var%':>8}\n")
                f.write("-" * 65 + "\n")
                
                for sample in sorted_samples:
                    res = joint_results.get(sample, {})
                    covered = res.get(f'{marker_name}_covered', 0)
                    variant = res.get(f'{marker_name}_variant', 0)
                    cov_pct = covered / marker_total * 100 if marker_total > 0 else 0
                    var_pct = variant / marker_total * 100 if marker_total > 0 else 0
                    f.write(f"{sample:<20} {covered:>12,} {cov_pct:>7.2f}% {variant:>12,} {var_pct:>7.2f}%\n")
                
                # Summary stats
                if joint_results:
                    avg_cov = sum(joint_results[s].get(f'{marker_name}_covered', 0) for s in sample_names) / len(sample_names)
                    avg_var = sum(joint_results[s].get(f'{marker_name}_variant', 0) for s in sample_names) / len(sample_names)
                    f.write("-" * 65 + "\n")
                    f.write(f"{'AVERAGE':<20} {avg_cov:>12,.1f} {avg_cov/marker_total*100:>7.2f}% {avg_var:>12,.1f} {avg_var/marker_total*100:>7.2f}%\n")
        
        print(f"  Written: {report2}")
    
    # Report 3: Summary comparison (GVCF vs Joint)
    if gvcf_results and joint_results:
        report3 = OUT_DIR / "coverage_comparison_gvcf_vs_joint.txt"
        with open(report3, 'w') as f:
            f.write("=" * 120 + "\n")
            f.write("Coverage Comparison: Individual GVCF vs Joint VCF\n")
            f.write("=" * 120 + "\n\n")
            
            for marker_name in sorted(marker_sets.keys()):
                marker_total = len(marker_sets[marker_name])
                f.write(f"\n{'='*90}\n")
                f.write(f"Marker Set: {marker_name} ({marker_total:,} markers)\n")
                f.write(f"{'='*90}\n")
                f.write(f"{'Sample':<20} {'GVCF_Cov':>10} {'GVCF_Cov%':>10} {'Joint_Cov':>10} {'Joint_Cov%':>10} {'Diff':>8}\n")
                f.write("-" * 75 + "\n")
                
                for gvcf_res in gvcf_results:
                    sample = gvcf_res['sample']
                    gvcf_cov = gvcf_res.get(f'{marker_name}_covered', 0)
                    gvcf_pct = gvcf_cov / marker_total * 100 if marker_total > 0 else 0
                    
                    joint_res = joint_results.get(sample, {})
                    joint_cov = joint_res.get(f'{marker_name}_covered', 0)
                    joint_pct = joint_cov / marker_total * 100 if marker_total > 0 else 0
                    
                    diff = joint_cov - gvcf_cov
                    
                    f.write(f"{sample:<20} {gvcf_cov:>10,} {gvcf_pct:>9.2f}% {joint_cov:>10,} {joint_pct:>9.2f}% {diff:>+8,}\n")
        
        print(f"  Written: {report3}")
    
    # Report 4: Marker set summary (all samples aggregated)
    report4 = OUT_DIR / "marker_set_summary.txt"
    with open(report4, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("Marker Set Coverage Summary (All Samples)\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"{'Marker Set':<15} {'Total':>10} ")
        if gvcf_results:
            f.write(f"{'GVCF_AvgCov':>12} {'GVCF_AvgCov%':>12} ")
        if joint_results:
            f.write(f"{'Joint_AvgCov':>12} {'Joint_AvgCov%':>12}")
        f.write("\n")
        f.write("-" * 90 + "\n")
        
        for marker_name in sorted(marker_sets.keys()):
            marker_total = len(marker_sets[marker_name])
            f.write(f"{marker_name:<15} {marker_total:>10,} ")
            
            if gvcf_results:
                avg_cov = sum(r.get(f'{marker_name}_covered', 0) for r in gvcf_results) / len(gvcf_results)
                avg_pct = avg_cov / marker_total * 100 if marker_total > 0 else 0
                f.write(f"{avg_cov:>12,.1f} {avg_pct:>11.2f}% ")
            
            if joint_results:
                avg_cov = sum(joint_results[s].get(f'{marker_name}_covered', 0) for s in sample_names) / len(sample_names)
                avg_pct = avg_cov / marker_total * 100 if marker_total > 0 else 0
                f.write(f"{avg_cov:>12,.1f} {avg_pct:>11.2f}%")
            
            f.write("\n")
    
    print(f"  Written: {report4}")
    
    # ========================================
    # Print summary to screen
    # ========================================
    print("\n" + "=" * 100)
    print("COVERAGE SUMMARY")
    print("=" * 100)
    
    print(f"\n{'Marker Set':<15} {'Total':>10}", end="")
    if gvcf_results:
        print(f" {'GVCF_AvgCov%':>12}", end="")
    if joint_results:
        print(f" {'Joint_AvgCov%':>13}", end="")
    print()
    print("-" * 60)
    
    for marker_name in sorted(marker_sets.keys()):
        marker_total = len(marker_sets[marker_name])
        print(f"{marker_name:<15} {marker_total:>10,}", end="")
        
        if gvcf_results:
            avg_cov = sum(r.get(f'{marker_name}_covered', 0) for r in gvcf_results) / len(gvcf_results)
            avg_pct = avg_cov / marker_total * 100 if marker_total > 0 else 0
            print(f" {avg_pct:>11.2f}%", end="")
        
        if joint_results:
            avg_cov = sum(joint_results[s].get(f'{marker_name}_covered', 0) for s in sample_names) / len(sample_names)
            avg_pct = avg_cov / marker_total * 100 if marker_total > 0 else 0
            print(f" {avg_pct:>12.2f}%", end="")
        
        print()
    
    print(f"\nReports saved to: {OUT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()