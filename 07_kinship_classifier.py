#!/usr/bin/env python3
"""
Kinship Relationship Classifier & Evaluator (Step 7) - v5
==========================================================
Per-metric classifier with ERROR-MINIMIZING thresholds:
  - Each metric (IBS, IBD, Kinship) classifies independently
  - Thresholds optimized to minimize total misclassifications
    (not simple midpoint between medians)
  - 2nd degree split into Sibling vs GP-GC
  - Grand-Uncle-Nephew = 4th degree

Usage:
  python 07_kinship_classifier.py --results-dir /path/to/06_kinship_analysis
  python 07_kinship_classifier.py --combined-csv all_results_combined.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['font.family'] = 'DejaVu Sans'

# ============================================================
# Constants
# ============================================================
METRICS = ['IBS', 'IBD', 'Kinship']

RELATIONSHIP_TO_DEGREE = {
    'Unrelated': 0, 'Spouse': 0,
    'Parent-Child': 1,
    'Sibling': 2, 'Grandparent-Grandchild': 2, 'Half-Sibling': 2,
    'Uncle-Nephew': 3, 'Great-Grandparent': 3,
    'Cousin': 4, 'Grand-Uncle-Nephew': 4,
    'Cousin-Once-Removed': 5,
    'Second-Cousin': 6,
}

RELATIONSHIP_TO_GROUP = {
    'Unrelated': 'G0_Unrelated', 'Spouse': 'G0_Unrelated',
    'Parent-Child': 'G1_1st',
    'Sibling': 'G2a_Sib', 'Half-Sibling': 'G2a_Sib',
    'Grandparent-Grandchild': 'G2b_GPGC',
    'Uncle-Nephew': 'G3_3rd', 'Great-Grandparent': 'G3_3rd',
    'Cousin': 'G4_4th', 'Grand-Uncle-Nephew': 'G4_4th',
    'Cousin-Once-Removed': 'G5_5th',
    'Second-Cousin': 'G6_6th',
}

GROUP_ORDER = ['G0_Unrelated', 'G1_1st', 'G2a_Sib', 'G2b_GPGC',
               'G3_3rd', 'G4_4th', 'G5_5th', 'G6_6th']

GROUP_DISPLAY = {
    'G0_Unrelated': 'Unrelated', 'G1_1st': '1st degree',
    'G2a_Sib': '2nd (Sibling)', 'G2b_GPGC': '2nd (GP-GC)',
    'G3_3rd': '3rd degree', 'G4_4th': '4th degree',
    'G5_5th': '5th degree', 'G6_6th': '6th degree',
}
GROUP_DISPLAY_SHORT = {
    'G0_Unrelated': 'Unrel', 'G1_1st': '1st',
    'G2a_Sib': '2nd-Sib', 'G2b_GPGC': '2nd-GPGC',
    'G3_3rd': '3rd', 'G4_4th': '4th',
    'G5_5th': '5th', 'G6_6th': '6th',
}
GROUP_TO_DEGREE = {
    'G0_Unrelated': 0, 'G1_1st': 1, 'G2a_Sib': 2, 'G2b_GPGC': 2,
    'G3_3rd': 3, 'G4_4th': 4, 'G5_5th': 5, 'G6_6th': 6
}

MARKER_COLORS = {
    'NFS_36K': '#1a5276', 'NFS_24K': '#2874a6', 'NFS_20K': '#3498db',
    'NFS_12K': '#e74c3c', 'NFS_6K': '#9b59b6',
    'Kintellignece': '#27ae60', 'Qiaseq': '#f39c12'
}
GROUP_COLORS = {
    'G0_Unrelated': '#bdc3c7', 'G1_1st': '#c0392b',
    'G2a_Sib': '#e74c3c', 'G2b_GPGC': '#f39c12',
    'G3_3rd': '#e67e22', 'G4_4th': '#f1c40f',
    'G5_5th': '#2ecc71', 'G6_6th': '#3498db'
}
METRIC_COLORS = {'IBS': '#3498db', 'IBD': '#e74c3c', 'Kinship': '#2ecc71'}


def is_nocancer(ms):
    return 'nocancer' in ms.lower()

def _gd(g):
    return GROUP_DISPLAY.get(g, g)

def _gs(g):
    return GROUP_DISPLAY_SHORT.get(g, g)


# ============================================================
# 0. Data Prep
# ============================================================
def fix_degree_labels(df):
    corrections = 0
    for rel, cdeg in RELATIONSHIP_TO_DEGREE.items():
        mask = (df['Relationship'] == rel) & (df['Degree'] != cdeg)
        n = mask.sum()
        if n > 0:
            print(f"    FIX: {rel} degree {df.loc[mask, 'Degree'].unique()} -> {cdeg} ({n} pairs)")
            df.loc[mask, 'Degree'] = cdeg
            corrections += n
    print(f"    Corrections: {corrections}" if corrections else "    No corrections needed.")
    df['Group'] = df['Relationship'].map(RELATIONSHIP_TO_GROUP)
    return df


def filter_real(df):
    return df[~(df['Sample1'].str.contains('GP|GM', na=False) |
                df['Sample2'].str.contains('GP|GM', na=False))].copy()


# ============================================================
# 1. Error-Minimizing Threshold Classifier
# ============================================================
def compute_group_stats(df_marker, metric):
    stats = {}
    for grp in df_marker['Group'].dropna().unique():
        v = df_marker[df_marker['Group'] == grp][metric].dropna()
        if len(v) == 0:
            continue
        stats[grp] = dict(
            n=len(v), mean=v.mean(), median=v.median(), std=v.std(),
            min=v.min(), max=v.max(),
            q10=v.quantile(.10), q25=v.quantile(.25),
            q75=v.quantile(.75), q90=v.quantile(.90))
    return stats


def find_optimal_boundary(valsA, valsB, medA, medB):
    """Find threshold between two groups that minimizes total misclassification.
    Assumes group A has higher metric values than group B (medA > medB).
    Threshold: values >= threshold -> A, values < threshold -> B.
    Returns: (optimal_threshold, midpoint_threshold, errors_optimal, errors_midpoint)
    """
    midpoint = (medA + medB) / 2
    all_vals = np.sort(np.concatenate([valsA, valsB]))
    candidates = [(all_vals[i] + all_vals[i + 1]) / 2
                  for i in range(len(all_vals) - 1)]
    candidates.append(midpoint)

    best_th, best_err = midpoint, len(valsA) + len(valsB)
    for th in candidates:
        errA = (valsA < th).sum()
        errB = (valsB >= th).sum()
        total = errA + errB
        if total < best_err:
            best_err, best_th = total, th

    mid_errA = (valsA < midpoint).sum()
    mid_errB = (valsB >= midpoint).sum()
    mid_err = mid_errA + mid_errB

    return best_th, midpoint, best_err, mid_err


def build_boundaries_optimized(df_marker, metric, gstats):
    """Build error-minimizing boundaries between adjacent groups."""
    medians = {g: s['median'] for g, s in gstats.items()}
    ordered = sorted(medians, key=lambda g: medians[g], reverse=True)

    pair_thresholds = {}
    total_saved = 0
    for i in range(len(ordered) - 1):
        gHi, gLo = ordered[i], ordered[i + 1]
        vHi = df_marker[df_marker['Group'] == gHi][metric].dropna().values
        vLo = df_marker[df_marker['Group'] == gLo][metric].dropna().values
        if len(vHi) == 0 or len(vLo) == 0:
            pair_thresholds[(gHi, gLo)] = (medians[gHi] + medians[gLo]) / 2
            continue

        opt_th, mid_th, opt_err, mid_err = find_optimal_boundary(
            vHi, vLo, medians[gHi], medians[gLo])
        pair_thresholds[(gHi, gLo)] = opt_th
        saved = mid_err - opt_err
        if saved > 0:
            total_saved += saved
            print(f"      {_gs(gHi)} vs {_gs(gLo)}: midpoint={mid_th:.5f}(err={mid_err}) "
                  f"-> optimal={opt_th:.5f}(err={opt_err}), saved={saved}")

    bounds = {}
    for i, grp in enumerate(ordered):
        hi = float('inf') if i == 0 else pair_thresholds[(ordered[i - 1], grp)]
        lo = float('-inf') if i == len(ordered) - 1 else pair_thresholds[(grp, ordered[i + 1])]
        bounds[grp] = (lo, hi)

    return bounds, pair_thresholds, total_saved


def classify_value(val, bounds):
    if pd.isna(val):
        return None
    for grp, (lo, hi) in bounds.items():
        if lo <= val < hi:
            return grp
    best, bestd = list(bounds.keys())[0], float('inf')
    for grp, (lo, hi) in bounds.items():
        mid = lo if hi == float('inf') else hi if lo == float('-inf') else (lo + hi) / 2
        d = abs(val - mid)
        if d < bestd:
            bestd, best = d, grp
    return best


# ============================================================
# 2. Run
# ============================================================
def run_classifier(all_df, marker_list):
    all_results = []
    cinfo = {}

    for ms in marker_list:
        print(f"\n  [{ms}]")
        df = filter_real(all_df[all_df['Marker_Set'] == ms])
        cinfo[ms] = {}

        for metric in METRICS:
            if metric not in df.columns:
                continue
            gstats = compute_group_stats(df, metric)
            if not gstats:
                continue

            print(f"    {metric}:")
            bounds, pair_th, saved = build_boundaries_optimized(df, metric, gstats)
            cinfo[ms][metric] = dict(stats=gstats, boundaries=bounds,
                                     pair_thresholds=pair_th, saved=saved)

            col_pred = f'Pred_{metric}'
            df[col_pred] = df[metric].apply(lambda v: classify_value(v, bounds))
            df[f'PredDeg_{metric}'] = df[col_pred].map(GROUP_TO_DEGREE)
            df[f'GroupCorrect_{metric}'] = df['Group'] == df[col_pred]
            df[f'DegCorrect_{metric}'] = df['Degree'] == df[f'PredDeg_{metric}']
            df[f'DegWithin1_{metric}'] = (df['Degree'] - df[f'PredDeg_{metric}']).abs() <= 1

            rel = df[df['Degree'] > 0]
            ga = rel[f'GroupCorrect_{metric}'].mean() * 100 if len(rel) else 0
            da = rel[f'DegCorrect_{metric}'].mean() * 100 if len(rel) else 0
            print(f"      => GroupAcc={ga:.1f}% DegreeAcc={da:.1f}% (saved {saved} errors)")

        all_results.append(df)

    return pd.concat(all_results, ignore_index=True), cinfo


# ============================================================
# 3. Evaluation
# ============================================================
def compute_group_accuracy(rdf, ml, metric):
    rows = []
    pred_col = f'Pred_{metric}'
    gc_col = f'GroupCorrect_{metric}'
    dc_col = f'DegCorrect_{metric}'
    w1_col = f'DegWithin1_{metric}'
    for ms in ml:
        df = rdf[rdf['Marker_Set'] == ms].dropna(subset=[pred_col])
        for grp in sorted(df['Group'].unique()):
            s = df[df['Group'] == grp]
            n = len(s)
            rows.append(dict(
                Marker_Set=ms, Group=grp, Label=_gd(grp),
                Degree=GROUP_TO_DEGREE.get(grp, -1),
                N=n, Correct=int(s[gc_col].sum()),
                GroupAcc=s[gc_col].sum() / n * 100 if n else 0,
                DegreeAcc=s[dc_col].sum() / n * 100 if n else 0,
                Within1=s[w1_col].sum() / n * 100 if n else 0))
    return pd.DataFrame(rows)


def compute_rel_accuracy(rdf, ml, metric):
    rows = []
    pred_col = f'Pred_{metric}'
    gc_col = f'GroupCorrect_{metric}'
    dc_col = f'DegCorrect_{metric}'
    for ms in ml:
        df = rdf[rdf['Marker_Set'] == ms].dropna(subset=[pred_col])
        for rel in df['Relationship'].unique():
            s = df[df['Relationship'] == rel]
            n = len(s)
            pm = s[pred_col].mode()
            rows.append(dict(
                Marker_Set=ms, Relationship=rel,
                True_Degree=int(s['Degree'].iloc[0]),
                True_Group=s['Group'].iloc[0],
                N=n,
                GroupAcc=s[gc_col].sum() / n * 100 if n else 0,
                DegreeAcc=s[dc_col].sum() / n * 100 if n else 0,
                MostPredGroup=pm.iloc[0] if len(pm) else '?'))
    return pd.DataFrame(rows)


def find_misclassified(rdf, ml, metric):
    rows = []
    pred_col = f'Pred_{metric}'
    gc_col = f'GroupCorrect_{metric}'
    for ms in ml:
        df = rdf[rdf['Marker_Set'] == ms].dropna(subset=[pred_col])
        mc = df[(df['Degree'] > 0) & (~df[gc_col])]
        for _, r in mc.iterrows():
            rows.append(dict(
                Marker_Set=ms, Family=r.get('Family1', '?'),
                Sample1=r['Sample1'], Sample2=r['Sample2'],
                Member1=r.get('Member1', ''), Member2=r.get('Member2', ''),
                True_Relationship=r['Relationship'],
                True_Group=r['Group'], True_Degree=int(r['Degree']),
                Pred_Group=r[pred_col],
                Pred_Degree=int(GROUP_TO_DEGREE.get(r[pred_col], -1)),
                Metric_Value=round(r[metric], 6) if pd.notna(r.get(metric)) else None))
    return pd.DataFrame(rows)


# ============================================================
# 4. Threshold Export
# ============================================================
def export_thresholds(cinfo, outdir):
    tdir = outdir / "thresholds"
    tdir.mkdir(parents=True, exist_ok=True)

    for ms, minfo in cinfo.items():
        fp = tdir / f"thresholds_{ms}.txt"
        with open(fp, 'w') as f:
            f.write(f"{'=' * 90}\n")
            f.write(f"CLASSIFICATION THRESHOLDS: {ms}\n")
            f.write(f"{'=' * 90}\n\n")
            f.write("Method: Error-minimizing thresholds (optimized per adjacent group pair)\n")
            f.write("2nd degree split: Sibling vs GP-GC (separate bins)\n")
            f.write("Each metric classifies independently\n\n")

            for metric in METRICS:
                if metric not in minfo:
                    continue
                bd = minfo[metric]['boundaries']
                gs = minfo[metric]['stats']
                saved = minfo[metric]['saved']

                f.write(f"{'~' * 90}\n")
                f.write(f"  METRIC: {metric}  (saved {saved} errors vs midpoint)\n")
                f.write(f"{'~' * 90}\n\n")

                f.write(f"  BOUNDARIES (error-minimizing)\n")
                f.write(f"  {'Group':<22} {'Lower':>12} {'Upper':>12}  |  "
                        f"{'Median':>10} {'Mean':>10} {'Std':>8} {'N':>5}\n")
                f.write(f"  " + "-" * 85 + "\n")
                for grp in GROUP_ORDER:
                    if grp not in bd:
                        continue
                    lo, hi = bd[grp]
                    s = gs.get(grp, {})
                    lo_s = f"{lo:.6f}" if lo != float('-inf') else "      -inf"
                    hi_s = f"{hi:.6f}" if hi != float('inf') else "      +inf"
                    f.write(f"  {_gd(grp):<22} {lo_s:>12} {hi_s:>12}  |  "
                            f"{s.get('median', 0):>10.6f} {s.get('mean', 0):>10.6f} "
                            f"{s.get('std', 0):>8.6f} {s.get('n', 0):>5}\n")

                f.write(f"\n  DETAILED STATISTICS\n")
                f.write(f"  {'Group':<22} {'N':>5} {'Min':>9} {'Q10':>9} {'Q25':>9} "
                        f"{'Median':>9} {'Q75':>9} {'Q90':>9} {'Max':>9}\n")
                f.write(f"  " + "-" * 90 + "\n")
                for grp in GROUP_ORDER:
                    if grp not in gs:
                        continue
                    s = gs[grp]
                    f.write(f"  {_gd(grp):<22} {s['n']:>5} "
                            f"{s['min']:>9.5f} {s['q10']:>9.5f} {s['q25']:>9.5f} "
                            f"{s['median']:>9.5f} {s['q75']:>9.5f} {s['q90']:>9.5f} "
                            f"{s['max']:>9.5f}\n")
                f.write("\n")
            f.write(f"{'=' * 90}\n")
        print(f"    Saved: thresholds_{ms}.txt")

        csv_rows = []
        for metric in METRICS:
            if metric not in minfo:
                continue
            bd = minfo[metric]['boundaries']
            gs = minfo[metric]['stats']
            for grp in GROUP_ORDER:
                if grp not in bd:
                    continue
                lo, hi = bd[grp]
                s = gs.get(grp, {})
                csv_rows.append(dict(
                    Metric=metric, Group=grp, GroupLabel=_gs(grp),
                    Degree=GROUP_TO_DEGREE.get(grp, -1),
                    Lower=lo if lo != float('-inf') else None,
                    Upper=hi if hi != float('inf') else None,
                    Median=s.get('median'), Mean=s.get('mean'),
                    Std=s.get('std'), N=s.get('n'),
                    Q25=s.get('q25'), Q75=s.get('q75')))
        pd.DataFrame(csv_rows).to_csv(tdir / f"thresholds_{ms}.csv", index=False)


# ============================================================
# 5. Plots
# ============================================================
def plot_accuracy_overall(rdf, marker_list, metric, fdir):
    all_no_nc = [m for m in marker_list if not is_nocancer(m)]
    nfs_only = [m for m in all_no_nc if m.startswith('NFS_')]
    pred_col = f'Pred_{metric}'
    gc_col = f'GroupCorrect_{metric}'

    def _summary(mlist):
        rows = []
        for ms in mlist:
            df = rdf[rdf['Marker_Set'] == ms].dropna(subset=[pred_col])
            n = len(df)
            nc = int(df[gc_col].sum())
            rel = df[df['Degree'] > 0]
            nr = len(rel)
            nrc = int(rel[gc_col].sum()) if nr else 0
            rows.append(dict(Marker_Set=ms,
                             All=nc / n * 100 if n else 0,
                             Related=nrc / nr * 100 if nr else 0))
        return pd.DataFrame(rows)

    for mlist_sub, suffix, subtitle in [
        (all_no_nc, 'all', 'All Marker Sets'),
        (nfs_only, 'nfs', 'NFS Only'),
    ]:
        if not mlist_sub:
            continue
        summ = _summary(mlist_sub)
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for ax, col, title in [
            (axes[0], 'All', f'[{metric}] All Pairs - {subtitle}'),
            (axes[1], 'Related', f'[{metric}] Related Only - {subtitle}'),
        ]:
            order = summ.sort_values(col, ascending=False)['Marker_Set'].tolist()
            vals = [summ[summ['Marker_Set'] == m][col].values[0] for m in order]
            colors = [MARKER_COLORS.get(m, '#95a5a6') for m in order]
            bars = ax.bar(range(len(order)), vals, color=colors, edgecolor='white')
            ax.set_xticks(range(len(order)))
            ax.set_xticklabels(order, rotation=45, ha='right', fontsize=10)
            ax.set_ylabel('Accuracy (%)')
            ax.set_title(title, fontweight='bold')
            ax.set_ylim(0, 105)
            ax.grid(axis='y', alpha=.3)
            for b in bars:
                ax.annotate(f'{b.get_height():.1f}%',
                            xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                            ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        fname = f"accuracy_overall_{metric}_{suffix}.png"
        plt.savefig(fdir / fname, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {fname}")


def plot_accuracy_heatmap_group(gadf, metric, path):
    markers = sorted(gadf['Marker_Set'].unique())
    groups = [g for g in GROUP_ORDER if g in gadf['Group'].values]
    pivot = gadf.pivot_table(index='Marker_Set', columns='Group',
                             values='GroupAcc', aggfunc='first')
    pivot = pivot.reindex(index=markers, columns=groups)

    fig, ax = plt.subplots(
        figsize=(max(12, len(groups) * 1.8), max(5, len(markers) * 0.8)))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', vmin=0, vmax=100,
                linewidths=.5, linecolor='white', ax=ax,
                cbar_kws={'label': 'Accuracy (%)', 'shrink': .8},
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    ax.set_xticklabels([_gd(g) for g in groups], rotation=25, ha='right', fontsize=9)
    ax.set_yticklabels(markers, rotation=0, fontsize=11)
    ax.set_xlabel('Classification Group')
    ax.set_ylabel('Marker Set')
    ax.set_title(f'[{metric}] Group Classification Accuracy',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path.name}")


def plot_accuracy_by_relationship(radf, metric, path):
    df = radf[radf['True_Degree'] > 0].copy()
    if len(df) == 0:
        return
    rel_order = ['Parent-Child', 'Sibling', 'Grandparent-Grandchild',
                 'Uncle-Nephew', 'Great-Grandparent', 'Cousin',
                 'Grand-Uncle-Nephew', 'Cousin-Once-Removed', 'Second-Cousin']
    rels = [r for r in rel_order if r in df['Relationship'].values]
    mlist = sorted(df['Marker_Set'].unique())

    fig, ax = plt.subplots(figsize=(max(14, len(rels) * 2), 7))
    nm = len(mlist)
    bw = 0.8 / nm
    x = np.arange(len(rels))
    for i, ms in enumerate(mlist):
        sub = df[df['Marker_Set'] == ms].set_index('Relationship')
        vals = [sub.loc[r, 'GroupAcc'] if r in sub.index else 0 for r in rels]
        bars = ax.bar(x + (i - nm / 2 + .5) * bw, vals, bw, label=ms,
                      color=MARKER_COLORS.get(ms, '#95a5a6'),
                      edgecolor='white', linewidth=.3)
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.annotate(f'{h:.0f}',
                            xy=(b.get_x() + b.get_width() / 2, h),
                            ha='center', va='bottom', fontsize=6,
                            fontweight='bold', rotation=90)
    ax.set_xticks(x)
    grpmap = {r: df[df['Relationship'] == r]['True_Group'].iloc[0] for r in rels}
    ax.set_xticklabels([f'{r}\n({_gs(grpmap.get(r, ""))})' for r in rels],
                       rotation=0, ha='center', fontsize=8)
    ax.set_ylabel('Group Classification Accuracy (%)')
    ax.set_title(f'[{metric}] Classification Accuracy by Relationship',
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, 120)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(axis='y', alpha=.3)
    ax.axhline(100, color='gray', ls='--', alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path.name}")


def plot_confusion_matrices(rdf, mlist, metric, outdir):
    pred_col = f'Pred_{metric}'
    for ms in mlist:
        df = rdf[rdf['Marker_Set'] == ms].dropna(subset=[pred_col])
        if len(df) == 0:
            continue
        yt = df['Group'].astype(str)
        yp = df[pred_col].astype(str)
        labels = [g for g in GROUP_ORDER if g in set(yt) | set(yp)]
        cm = confusion_matrix(yt, yp, labels=labels)
        cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
        tl = [_gd(l) for l in labels]

        fig, axes = plt.subplots(1, 2, figsize=(18, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=tl, yticklabels=tl,
                    linewidths=.5, linecolor='white')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        axes[0].set_title('Counts', fontweight='bold')
        plt.setp(axes[0].get_xticklabels(), rotation=30, ha='right', fontsize=8)
        plt.setp(axes[0].get_yticklabels(), rotation=0, fontsize=8)

        sns.heatmap(cmn, annot=True, fmt='.1f', cmap='RdYlGn',
                    vmin=0, vmax=100, ax=axes[1],
                    xticklabels=tl, yticklabels=tl,
                    linewidths=.5, linecolor='white')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        axes[1].set_title('Row-normalized (%)', fontweight='bold')
        plt.setp(axes[1].get_xticklabels(), rotation=30, ha='right', fontsize=8)
        plt.setp(axes[1].get_yticklabels(), rotation=0, fontsize=8)

        plt.suptitle(f'[{metric}] Confusion Matrix - {ms}',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(outdir / f"confusion_{metric}_{ms}.png",
                    dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: confusion_{metric}_{ms}.png")


def plot_misclassification_summary(mcdf, metric, path):
    if len(mcdf) == 0:
        print("    No misclassifications.")
        return
    mcdf_f = mcdf[~mcdf['Marker_Set'].apply(is_nocancer)]
    pivot = mcdf_f.groupby(['Marker_Set', 'True_Group']).size().reset_index(name='Errors')
    mlist = sorted(pivot['Marker_Set'].unique())
    groups = [g for g in GROUP_ORDER if g in pivot['True_Group'].values]

    fig, ax = plt.subplots(figsize=(max(12, len(groups) * 2), 6))
    nm = len(mlist)
    bw = 0.8 / nm
    x = np.arange(len(groups))
    for i, ms in enumerate(mlist):
        sub = pivot[pivot['Marker_Set'] == ms].set_index('True_Group')
        vals = [sub.loc[g, 'Errors'] if g in sub.index else 0 for g in groups]
        bars = ax.bar(x + (i - nm / 2 + .5) * bw, vals, bw, label=ms,
                      color=MARKER_COLORS.get(ms, '#95a5a6'),
                      edgecolor='white', linewidth=.3)
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.annotate(f'{int(h)}',
                            xy=(b.get_x() + b.get_width() / 2, h),
                            ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([_gd(g) for g in groups], fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('# Misclassified Pairs')
    ax.set_title(f'[{metric}] Misclassification by True Group (error-minimized)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path.name}")


def plot_forensic_scenarios(rdf, mlist, metric, path):
    mlist = [m for m in mlist if not is_nocancer(m)]
    gc_col = f'GroupCorrect_{metric}'
    w1_col = f'DegWithin1_{metric}'
    pred_col = f'Pred_{metric}'

    fig, axes = plt.subplots(2, 1, figsize=(16, 12))

    ax = axes[0]
    all_groups = [g for g in GROUP_ORDER if g != 'G0_Unrelated']
    for ms in mlist:
        df = rdf[(rdf['Marker_Set'] == ms) & (rdf['Degree'] > 0)].dropna(subset=[pred_col])
        accs = []
        for g in all_groups:
            sub = df[df['Group'] == g]
            accs.append(sub[gc_col].mean() * 100 if len(sub) else 0)
        ax.plot(range(len(all_groups)), accs, 'o-', label=ms,
                color=MARKER_COLORS.get(ms, 'gray'), lw=2, ms=8)
    ax.set_xticks(range(len(all_groups)))
    ax.set_xticklabels([_gd(g) for g in all_groups], rotation=15,
                       ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'[{metric}] Group Accuracy by Kinship Distance',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(0, 105)
    ax.grid(alpha=.3)

    ax = axes[1]
    da = []
    for ms in mlist:
        df = rdf[(rdf['Marker_Set'] == ms) &
                 (rdf['Degree'].isin([5, 6]))].dropna(subset=[pred_col])
        ex = df[gc_col].mean() * 100 if len(df) else 0
        w1 = df[w1_col].mean() * 100 if len(df) else 0
        da.append(dict(Marker_Set=ms, Exact=ex, Within1=w1))
    da = pd.DataFrame(da)
    x = np.arange(len(da))
    w = 0.35
    bars1 = ax.bar(x - w / 2, da['Exact'], w, label='Exact',
                   color='#3498db', edgecolor='white')
    bars2 = ax.bar(x + w / 2, da['Within1'], w, label='Within +/-1',
                   color='#2ecc71', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(da['Marker_Set'], rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'[{metric}] Extended Kinship (5-6th)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=.3)
    for b in list(bars1) + list(bars2):
        h = b.get_height()
        ax.annotate(f'{h:.0f}%',
                    xy=(b.get_x() + b.get_width() / 2, h),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path.name}")


def plot_thresholds(cinfo, ms, metric, path):
    """Threshold plot: red values on x-axis, no box."""
    if metric not in cinfo[ms]:
        return
    bd = cinfo[ms][metric]['boundaries']
    gs = cinfo[ms][metric]['stats']
    grps = [g for g in GROUP_ORDER if g in gs]

    fig, ax = plt.subplots(figsize=(10, max(5, len(grps) * 0.8)))

    for i, grp in enumerate(grps):
        s = gs[grp]
        c = GROUP_COLORS.get(grp, '#95a5a6')
        ax.barh(i, s['q75'] - s['q25'], left=s['q25'], height=.6,
                color=c, alpha=.6, edgecolor='black', lw=.5)
        ax.plot(s['median'], i, 'k|', ms=15, mew=2)
        ax.plot([s['min'], s['max']], [i, i], 'k-', lw=.5, alpha=.5)
        ax.annotate(f'{s["median"]:.4f}', xy=(s['median'], i + 0.35),
                    ha='center', va='bottom', fontsize=7,
                    color='black', fontweight='bold')

    drawn = set()
    for grp, (lo, hi) in bd.items():
        for th in [lo, hi]:
            if th not in (float('inf'), float('-inf')):
                th_r = round(th, 8)
                if th_r not in drawn:
                    ax.axvline(th, color='red', ls='--', alpha=.6, lw=1.2)
                    ax.annotate(f'{th:.4f}',
                                xy=(th, -0.7),
                                ha='center', va='top',
                                fontsize=7, color='red', fontweight='bold')
                    drawn.add(th_r)

    ax.set_yticks(range(len(grps)))
    ax.set_yticklabels([_gd(g) for g in grps], fontsize=9)
    ax.set_xlabel(metric, fontsize=12)
    ax.set_title(f'{ms} - {metric}\n(Box=IQR, |=median, red dashed=threshold [error-minimized])',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=.3)
    ax.set_ylim(-1.2, len(grps) - 0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path.name}")


def plot_metric_comparison(rdf, marker_list, path):
    mlist = [m for m in marker_list if not is_nocancer(m)]
    rows = []
    for ms in mlist:
        df = rdf[rdf['Marker_Set'] == ms]
        for metric in METRICS:
            gc_col = f'GroupCorrect_{metric}'
            if gc_col not in df.columns:
                continue
            valid = df.dropna(subset=[f'Pred_{metric}'])
            rel = valid[valid['Degree'] > 0]
            rows.append(dict(Marker_Set=ms, Metric=metric,
                             All=valid[gc_col].mean() * 100 if len(valid) else 0,
                             Related=rel[gc_col].mean() * 100 if len(rel) else 0))
    mdf = pd.DataFrame(rows)
    if len(mdf) == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, col, title in zip(axes, ['All', 'Related'],
                              ['All Pairs', 'Related Only']):
        pv = mdf.pivot_table(index='Marker_Set', columns='Metric', values=col)
        x = np.arange(len(pv))
        w = 0.25
        for i, m in enumerate(METRICS):
            if m in pv.columns:
                bars = ax.bar(x + (i - 1) * w, pv[m], w, label=m,
                              color=METRIC_COLORS[m], edgecolor='white')
                for b in bars:
                    h = b.get_height()
                    ax.annotate(f'{h:.1f}',
                                xy=(b.get_x() + b.get_width() / 2, h),
                                ha='center', va='bottom', fontsize=7,
                                fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(pv.index, rotation=45, ha='right')
        ax.set_ylabel('Group Accuracy (%)')
        ax.set_title(f'Metric Comparison ({title})', fontweight='bold')
        ax.legend()
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path.name}")


# ============================================================
# 6. Master Tables & Report
# ============================================================
def generate_master_tables(rdf, mlist, tdir):
    for ms in mlist:
        df = rdf[rdf['Marker_Set'] == ms].copy()
        cols = ['Sample1', 'Sample2', 'Family1', 'Relationship', 'Degree', 'Group',
                'IBS', 'IBD', 'Kinship']
        for m in METRICS:
            cols += [f'Pred_{m}', f'PredDeg_{m}',
                     f'GroupCorrect_{m}', f'DegCorrect_{m}']
        avail = [c for c in cols if c in df.columns]
        df[avail].sort_values(
            ['Degree', 'Family1', 'Sample1', 'Sample2'],
            ascending=[False, True, True, True]
        ).to_csv(tdir / f"master_table_{ms}.csv", index=False)
        print(f"    Saved: master_table_{ms}.csv")


def generate_report(rdf, all_gadf, all_radf, all_mcdf, cinfo, mlist, rpath):
    with open(rpath, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("KINSHIP CLASSIFIER - EVALUATION REPORT (v5)\n")
        f.write("=" * 100 + "\n\n")
        f.write("METHOD\n" + "-" * 80 + "\n")
        f.write("  Per-metric classification (IBS, IBD, Kinship independently)\n")
        f.write("  Error-minimizing thresholds (not simple midpoint)\n")
        f.write("  2nd degree split: Sibling vs GP-GC (separate bins)\n")
        f.write("  Grand-Uncle-Nephew = 4th degree\n\n")

        for metric in METRICS:
            gadf = all_gadf[metric]
            radf = all_radf[metric]
            mcdf = all_mcdf[metric]

            f.write(f"\n{'#' * 100}\n")
            f.write(f"  METRIC: {metric}\n")
            f.write(f"{'#' * 100}\n\n")

            f.write(f"  1. SUMMARY\n  " + "-" * 70 + "\n\n")
            for ms in mlist:
                gc_col = f'GroupCorrect_{metric}'
                dc_col = f'DegCorrect_{metric}'
                df = rdf[rdf['Marker_Set'] == ms].dropna(subset=[f'Pred_{metric}'])
                n = len(df)
                nc = int(df[gc_col].sum())
                ndc = int(df[dc_col].sum())
                rel = df[df['Degree'] > 0]
                nr = len(rel)
                nrc = int(rel[gc_col].sum()) if nr else 0
                nrdc = int(rel[dc_col].sum()) if nr else 0
                saved = cinfo.get(ms, {}).get(metric, {}).get('saved', 0)
                f.write(f"    [{ms}]  (saved {saved} errors vs midpoint)\n")
                f.write(f"      All:     {n:>6} | GroupAcc: {nc / n * 100:>5.1f}% | "
                        f"DegreeAcc: {ndc / n * 100:>5.1f}%\n")
                if nr:
                    f.write(f"      Related: {nr:>6} | GroupAcc: {nrc / nr * 100:>5.1f}% | "
                            f"DegreeAcc: {nrdc / nr * 100:>5.1f}%\n")
                f.write("\n")

            f.write(f"\n  2. PER-GROUP ACCURACY\n  " + "-" * 70 + "\n")
            for ms in mlist:
                f.write(f"\n    [{ms}]\n")
                f.write(f"    {'Group':<22} {'N':>5} {'GrpAcc':>7} "
                        f"{'DegAcc':>7} {'+/-1':>6}\n")
                f.write(f"    " + "-" * 50 + "\n")
                for _, r in gadf[gadf['Marker_Set'] == ms].sort_values('Group').iterrows():
                    f.write(f"    {r['Label']:<22} {int(r['N']):>5} "
                            f"{r['GroupAcc']:>6.1f}% {r['DegreeAcc']:>6.1f}% "
                            f"{r['Within1']:>5.1f}%\n")

            f.write(f"\n\n  3. PER-RELATIONSHIP ACCURACY\n  " + "-" * 70 + "\n")
            for ms in mlist:
                f.write(f"\n    [{ms}]\n")
                sub = radf[(radf['Marker_Set'] == ms) &
                           (radf['True_Degree'] > 0)].sort_values('True_Group')
                f.write(f"    {'Relationship':<26} {'Group':>10} {'N':>5} "
                        f"{'GrpAcc':>7} {'DegAcc':>7} {'->':>10}\n")
                f.write(f"    " + "-" * 70 + "\n")
                for _, r in sub.iterrows():
                    f.write(f"    {r['Relationship']:<26} "
                            f"{_gs(r['True_Group']):>10} {int(r['N']):>5} "
                            f"{r['GroupAcc']:>6.1f}% {r['DegreeAcc']:>6.1f}% "
                            f"->{_gs(r['MostPredGroup']):>9}\n")

            f.write(f"\n\n  4. MISCLASSIFIED (related)\n  " + "-" * 70 + "\n")
            if len(mcdf) == 0:
                f.write("    None!\n")
            else:
                for ms in mlist:
                    sub = mcdf[mcdf['Marker_Set'] == ms]
                    if len(sub) == 0:
                        f.write(f"\n    [{ms}] None\n")
                        continue
                    f.write(f"\n    [{ms}] {len(sub)} misclassified\n")
                    f.write(f"    {'Fam':>4} {'S1':<16} {'S2':<16} "
                            f"{'Relation':<22} {'True':>10} {'Pred':>10} "
                            f"{metric:>10}\n")
                    f.write(f"    " + "-" * 95 + "\n")
                    for _, r in sub.sort_values(['True_Group', 'Family']).iterrows():
                        mv = (f"{r['Metric_Value']:.5f}"
                              if pd.notna(r.get('Metric_Value')) else 'N/A')
                        f.write(f"    {r['Family']:>4} "
                                f"{str(r['Sample1']):<16} "
                                f"{str(r['Sample2']):<16} "
                                f"{r['True_Relationship']:<22} "
                                f"{_gs(r['True_Group']):>10} -> "
                                f"{_gs(r['Pred_Group']):>8} {mv:>10}\n")

        f.write("\n\n" + "=" * 100 + "\nEND OF REPORT\n" + "=" * 100 + "\n")
    print(f"    Saved: {rpath.name}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Kinship Classifier v5 (per-metric, error-minimizing)')
    parser.add_argument('--results-dir', type=str)
    parser.add_argument('--combined-csv', type=str)
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    if args.combined_csv:
        cpath = Path(args.combined_csv)
    elif args.results_dir:
        cpath = Path(args.results_dir) / "all_results_combined.csv"
    else:
        cpath = (Path.home() / "kinship/Analysis/20251031_wgrs"
                 / "06_kinship_analysis/all_results_combined.csv")

    if not cpath.exists():
        print(f"ERROR: {cpath} not found")
        sys.exit(1)

    print("=" * 70)
    print("KINSHIP CLASSIFIER v5 (per-metric, error-minimizing thresholds)")
    print("=" * 70)
    print(f"\n[1] Loading: {cpath}")
    all_df = pd.read_csv(cpath)
    marker_list = sorted(all_df['Marker_Set'].unique())
    ml = [m for m in marker_list if not is_nocancer(m)]
    print(f"    {len(all_df):,} rows, markers: {marker_list}")
    excluded = [m for m in marker_list if is_nocancer(m)]
    if excluded:
        print(f"    Excluding nocancer: {excluded}")

    print(f"\n[1b] Fixing degrees & adding groups...")
    all_df = fix_degree_labels(all_df)

    outdir = (Path(args.output_dir) if args.output_dir else
              Path(args.results_dir) / "classifier" if args.results_dir else
              cpath.parent / "classifier")
    fdir, tdir = outdir / "figures", outdir / "tables"
    for d in [outdir, fdir, tdir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n[2] Classifying (per-metric, error-minimizing)...")
    rdf, cinfo = run_classifier(all_df, ml)

    print(f"\n[3] Evaluating...")
    all_gadf, all_radf, all_mcdf = {}, {}, {}
    for metric in METRICS:
        print(f"\n  --- {metric} ---")
        gadf = compute_group_accuracy(rdf, ml, metric)
        gadf.to_csv(tdir / f"accuracy_group_{metric}.csv", index=False)
        radf = compute_rel_accuracy(rdf, ml, metric)
        radf.to_csv(tdir / f"accuracy_rel_{metric}.csv", index=False)
        mcdf = find_misclassified(rdf, ml, metric)
        mcdf.to_csv(tdir / f"misclassified_{metric}.csv", index=False)
        print(f"    Misclassified: {len(mcdf)}")
        all_gadf[metric] = gadf
        all_radf[metric] = radf
        all_mcdf[metric] = mcdf

    print(f"\n[4] Thresholds...")
    export_thresholds(cinfo, outdir)

    print(f"\n[5] Master tables...")
    generate_master_tables(rdf, ml, tdir)

    print(f"\n[6] Figures...")
    for metric in METRICS:
        print(f"\n  --- {metric} ---")
        plot_accuracy_overall(rdf, ml, metric, fdir)
        plot_accuracy_heatmap_group(all_gadf[metric], metric,
                                   fdir / f"heatmap_group_{metric}.png")
        plot_accuracy_by_relationship(all_radf[metric], metric,
                                     fdir / f"accuracy_rel_{metric}.png")
        plot_confusion_matrices(rdf, ml, metric, fdir)
        plot_misclassification_summary(all_mcdf[metric], metric,
                                      fdir / f"misclass_{metric}.png")
        plot_forensic_scenarios(rdf, ml, metric,
                               fdir / f"forensic_{metric}.png")
        for ms in ml:
            plot_thresholds(cinfo, ms, metric,
                            fdir / f"thresholds_{metric}_{ms}.png")

    print(f"\n  --- Metric comparison ---")
    plot_metric_comparison(rdf, ml, fdir / "metric_comparison.png")

    print(f"\n[7] Report...")
    generate_report(rdf, all_gadf, all_radf, all_mcdf, cinfo, ml,
                    outdir / "classifier_report.txt")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nOutput: {outdir}/")

    print("\n--- QUICK SUMMARY ---")
    for metric in METRICS:
        print(f"\n  [{metric}]")
        for ms in ml:
            gc = f'GroupCorrect_{metric}'
            df = rdf[rdf['Marker_Set'] == ms].dropna(subset=[f'Pred_{metric}'])
            rel = df[df['Degree'] > 0]
            if len(rel):
                ga = rel[gc].mean() * 100
                dist = rel[rel['Degree'].isin([5, 6])]
                dga = dist[gc].mean() * 100 if len(dist) else 0
                saved = cinfo.get(ms, {}).get(metric, {}).get('saved', 0)
                print(f"    {ms:<15}: GroupAcc={ga:.1f}%  "
                      f"Distant(5-6th)={dga:.1f}%  saved={saved}")


if __name__ == '__main__':
    main()
