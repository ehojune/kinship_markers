#!/usr/bin/env python3
"""
Step 0: Generate BED files for each marker set
- NFS markers: 36K, 24K, 20K, 12K, 6K (from TSV)
- External: Kintelligence, QIAseq
- Merged: All combined

No external dependencies (bedtools not required)
"""

from pathlib import Path

# ============================================================
# Configuration
# ============================================================
HOME = Path.home()
KINSHIP_DIR = Path("/BiO/Access/ehojune/kinship")
WORK_DIR = HOME / "kinship/Analysis/20251031_wgrs"
OUT_DIR = WORK_DIR / "06_kinship_analysis/marker_beds"

# Input files
MARKER_TSV = KINSHIP_DIR / "kinship_marker_with_20K_with_cancerfiltering.tsv"

# External marker sets
KINTELLIGENCE_BED = KINSHIP_DIR / "kintelligence_converted_to_chrpos.rightposition.bed"
QIASEQ_BED = KINSHIP_DIR / "QIAseq_DNA_panel_hg38.rightposition.bed"

# Output
MERGED_BED = KINSHIP_DIR / "merged_markers.bed"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def chrom_sort_key(chrom):
    """Natural sorting for chromosomes"""
    chrom = str(chrom).replace('chr', '')
    if chrom.isdigit():
        return (0, int(chrom), '')
    elif chrom == 'X':
        return (1, 0, 'X')
    elif chrom == 'Y':
        return (1, 1, 'Y')
    elif chrom == 'M' or chrom == 'MT':
        return (1, 2, 'M')
    else:
        return (2, 0, chrom)


def read_bed_regions(filepath):
    """Read BED file and return set of (chrom, start, end) tuples"""
    regions = set()
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                chrom = parts[0]
                try:
                    start = int(parts[1])
                    end = int(parts[2])
                    regions.add((chrom, start, end))
                except ValueError:
                    continue
    return regions


def write_bed(regions, output_path):
    """Write sorted regions to BED file"""
    sorted_regions = sorted(regions, key=lambda x: (chrom_sort_key(x[0]), x[1], x[2]))
    with open(output_path, 'w') as f:
        for chrom, start, end in sorted_regions:
            f.write(f"{chrom}\t{start}\t{end}\n")
    return len(sorted_regions)


def generate_nfs_beds():
    """Generate NFS marker BED files from TSV"""
    
    print("\n[1] Generating NFS marker BED files from TSV...")
    
    if not MARKER_TSV.exists():
        print(f"  ERROR: TSV not found: {MARKER_TSV}")
        return {}
    
    # Read TSV
    import csv
    markers = []
    with open(MARKER_TSV, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            markers.append(row)
    
    print(f"  Total markers in TSV: {len(markers):,}")
    
    # Marker sets to extract
    marker_sets = ['36K', '24K', '20K', '12K', '6K']
    nfs_beds = {}
    
    for marker_set in marker_sets:
        regions = set()
        for row in markers:
            if row.get(marker_set) == 'O':
                chrom = row['CHROM']
                pos = int(row['POS'])
                # BED format: 0-based start, 1-based end
                regions.add((chrom, pos - 1, pos))
        
        out_file = OUT_DIR / f"NFS_{marker_set}.bed"
        count = write_bed(regions, out_file)
        nfs_beds[f"NFS_{marker_set}"] = out_file
        print(f"  NFS_{marker_set}: {count:,} markers -> {out_file.name}")
    
    return nfs_beds


def copy_external_beds():
    """Copy/link external marker BED files"""
    
    print("\n[2] Processing external marker sets...")
    
    external_beds = {}
    
    for name, src_path in [('Kintelligence', KINTELLIGENCE_BED), ('QIAseq', QIASEQ_BED)]:
        dst_path = OUT_DIR / f"{name}.bed"
        
        if not src_path.exists():
            print(f"  WARNING: {name} not found: {src_path}")
            continue
        
        # Copy content (not symlink for portability)
        regions = read_bed_regions(src_path)
        count = write_bed(regions, dst_path)
        external_beds[name] = dst_path
        print(f"  {name}: {count:,} markers -> {dst_path.name}")
    
    return external_beds


def create_merged_bed(all_beds):
    """Merge all marker BED files"""
    
    print("\n[3] Creating merged BED file...")
    
    all_regions = set()
    
    for name, path in all_beds.items():
        if path.exists():
            regions = read_bed_regions(path)
            print(f"  Loading {name}: {len(regions):,}")
            all_regions |= regions
    
    count = write_bed(all_regions, MERGED_BED)
    print(f"\n  Merged (union): {count:,} unique positions -> {MERGED_BED}")
    
    return MERGED_BED


def analyze_overlap(all_beds):
    """Analyze overlap between all marker sets"""
    
    print("\n[4] Analyzing marker set overlaps...")
    
    # Load all marker sets
    marker_sets = {}
    for name, path in all_beds.items():
        if path.exists():
            marker_sets[name] = read_bed_regions(path)
    
    names = list(marker_sets.keys())
    
    # Print sizes
    print("\n  --- Marker Set Sizes ---")
    for name in names:
        print(f"  {name:15}: {len(marker_sets[name]):>10,}")
    
    # Pairwise overlap matrix
    print("\n  --- Pairwise Overlap Matrix ---")
    
    # Header
    print(f"\n  {'':15}", end="")
    for name in names:
        short_name = name[:12]
        print(f"{short_name:>12}", end="")
    print()
    
    for name1 in names:
        print(f"  {name1:15}", end="")
        for name2 in names:
            overlap = len(marker_sets[name1] & marker_sets[name2])
            print(f"{overlap:>12,}", end="")
        print()
    
    # Pairwise percentages (what % of row is in column)
    print("\n  --- Overlap % (row contained in column) ---")
    print(f"\n  {'':15}", end="")
    for name in names:
        short_name = name[:12]
        print(f"{short_name:>12}", end="")
    print()
    
    for name1 in names:
        print(f"  {name1:15}", end="")
        total1 = len(marker_sets[name1])
        for name2 in names:
            overlap = len(marker_sets[name1] & marker_sets[name2])
            pct = (overlap / total1 * 100) if total1 > 0 else 0
            print(f"{pct:>11.1f}%", end="")
        print()
    
    # NFS subset relationships
    print("\n  --- NFS Marker Subset Relationships ---")
    nfs_names = [n for n in names if n.startswith('NFS_')]
    nfs_sizes = [(n, len(marker_sets[n])) for n in nfs_names]
    nfs_sizes.sort(key=lambda x: -x[1])  # Sort by size descending
    
    for i, (name1, size1) in enumerate(nfs_sizes):
        for name2, size2 in nfs_sizes[i+1:]:
            overlap = len(marker_sets[name1] & marker_sets[name2])
            if overlap == size2:
                print(f"  {name2} ⊂ {name1} (100% contained)")
            else:
                pct = (overlap / size2 * 100)
                print(f"  {name2} ∩ {name1}: {overlap:,} ({pct:.1f}% of {name2})")
    
    # External vs NFS
    print("\n  --- External vs NFS_36K ---")
    nfs36k = marker_sets.get('NFS_36K', set())
    for name in ['Kintelligence', 'QIAseq']:
        if name in marker_sets:
            ext_set = marker_sets[name]
            overlap = len(ext_set & nfs36k)
            pct_ext = (overlap / len(ext_set) * 100) if ext_set else 0
            pct_nfs = (overlap / len(nfs36k) * 100) if nfs36k else 0
            print(f"  {name} ∩ NFS_36K: {overlap:,} ({pct_ext:.1f}% of {name}, {pct_nfs:.1f}% of NFS_36K)")
    
    # Unique to external
    print("\n  --- Unique to External Sets ---")
    all_nfs = set()
    for name in nfs_names:
        all_nfs |= marker_sets[name]
    
    for name in ['Kintelligence', 'QIAseq']:
        if name in marker_sets:
            unique = marker_sets[name] - all_nfs
            pct = (len(unique) / len(marker_sets[name]) * 100) if marker_sets[name] else 0
            print(f"  {name} only: {len(unique):,} ({pct:.1f}%)")
    
    # Save report
    report_path = OUT_DIR / "marker_overlap_report.txt"
    with open(report_path, 'w') as f:
        f.write("Marker Set Overlap Report\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("Marker Set Sizes:\n")
        for name in names:
            f.write(f"  {name}: {len(marker_sets[name]):,}\n")
        
        f.write("\nPairwise Overlap Counts:\n")
        for name1 in names:
            for name2 in names:
                if name1 < name2:
                    overlap = len(marker_sets[name1] & marker_sets[name2])
                    f.write(f"  {name1} ∩ {name2}: {overlap:,}\n")
    
    print(f"\n  Report saved: {report_path}")


def main():
    print("=" * 60)
    print("Step 0: Generate Marker BED Files")
    print("=" * 60)
    
    all_beds = {}
    
    # Generate NFS BEDs
    nfs_beds = generate_nfs_beds()
    all_beds.update(nfs_beds)
    
    # Copy external BEDs
    external_beds = copy_external_beds()
    all_beds.update(external_beds)
    
    # Create merged BED
    create_merged_bed(all_beds)
    
    # Analyze overlap
    analyze_overlap(all_beds)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\n  Output directory: {OUT_DIR}")
    print(f"  Merged BED: {MERGED_BED}")
    print(f"\n  Generated BED files:")
    for name, path in sorted(all_beds.items()):
        if path.exists():
            count = sum(1 for _ in open(path))
            print(f"    {name:15}: {count:>10,} markers")
    
    print(f"\n  Next step: Run 01_run_haplotypecaller_gvcf.py")


if __name__ == "__main__":
    main()