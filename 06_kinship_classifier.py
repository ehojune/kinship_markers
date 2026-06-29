#!/usr/bin/env python3
"""
Kinship Relationship Classifier & Evaluator (Step 6) - v8
==========================================================
Per-metric classifier with ROC-based thresholds (Youden's J):
  - Each metric (IBS, IBD, Kinship) classifies independently
  - Adjacent group pairs: ROC curve -> Youden's J optimal threshold
  - Thresholds chained sequentially for multi-class classification
  - 2nd degree split into Sibling vs GP-GC
  - Grand-Uncle-Nephew = 4th degree

Usage:
  python 06_kinship_classifier.py --eval-dir /path/to/07_evaluate_kinship
  python 06_kinship_classifier.py --combined-csv all_results_combined.csv
"""

import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'axes.unicode_minus':False,'figure.dpi':150,
                     'figure.facecolor':'white','font.family':'DejaVu Sans'})

# Final manuscript figures must not render chart titles.
def _disable_plot_titles():
    def _noop_title(self, *args, **kwargs):
        return None
    Axes.set_title = _noop_title
    Figure.suptitle = _noop_title

_disable_plot_titles()

METRICS = ['IBS','IBD','Kinship']

DEFAULT_WORK_DIR = Path('/mnt/d/Research/20251031_wgrs')
DEFAULT_ANALYSIS_DIR = DEFAULT_WORK_DIR / '06_kinship_analysis'
DEFAULT_EVAL_SUBDIR = '07_evaluate_kinship'
DEFAULT_EVAL_DIR = DEFAULT_WORK_DIR / DEFAULT_EVAL_SUBDIR
DEFAULT_CLASSIFIER_SUBDIR = 'classifier'
COMBINED_CSV_NAME = 'all_results_combined.csv'
COMPARISON_MARKERS = ['NFS_36K', 'NFS_24K', 'NFS_12K', 'NFS_6K', 'Kintelligence', 'QIAseq']

RELATIONSHIP_TO_DEGREE = {
    'Unrelated':0,'Spouse':0,'In-Law':0,'InLaw':0,'Inlaw':0,'Spouse/InLaw':0,'Between-Fam':0,'Parent-Child':1,
    'Sibling':2,'Grandparent-Grandchild':2,'Half-Sibling':2,
    'Uncle-Nephew':3,'Great-Grandparent':3,
    'Cousin':4,'Grand-Uncle-Nephew':4,
    'Cousin-Once-Removed':5,'Second-Cousin':6,
}
RELATIONSHIP_TO_GROUP = {
    'Unrelated':'G0_Unrelated','Spouse':'G0_Unrelated','In-Law':'G0_Unrelated','InLaw':'G0_Unrelated','Inlaw':'G0_Unrelated','Spouse/InLaw':'G0_Unrelated','Between-Fam':'G0_Unrelated',
    'Parent-Child':'G1_1st',
    'Sibling':'G2a_Sib','Half-Sibling':'G2a_Sib',
    'Grandparent-Grandchild':'G2b_GPGC',
    'Uncle-Nephew':'G3_3rd','Great-Grandparent':'G3_3rd',
    'Cousin':'G4_4th','Grand-Uncle-Nephew':'G4_4th',
    'Cousin-Once-Removed':'G5_5th','Second-Cousin':'G6_6th',
}
GROUP_ORDER = ['G1_1st','G2a_Sib','G3_3rd','G2b_GPGC',
               'G4_4th','G5_5th','G6_6th','G0_Unrelated']
THRESHOLD_PLOT_ORDER = list(reversed(GROUP_ORDER))
GROUP_DISPLAY = {
    'G0_Unrelated':'unrelated','G1_1st':'1st',
    'G2a_Sib':'2nd(sibling)','G2b_GPGC':'2nd(GP-GC)',
    'G3_3rd':'3rd','G4_4th':'4th',
    'G5_5th':'5th','G6_6th':'6th',
}
GROUP_DISPLAY_SHORT = {
    'G0_Unrelated':'unrelated','G1_1st':'1st',
    'G2a_Sib':'2nd(sibling)','G2b_GPGC':'2nd(GP-GC)',
    'G3_3rd':'3rd','G4_4th':'4th','G5_5th':'5th','G6_6th':'6th',
}
GROUP_DISPLAY_WITH_RELATIONSHIP = {
    'G0_Unrelated':'unrelated\n(Unrelated)',
    'G1_1st':'1st\n(Parent-Child)',
    'G2a_Sib':'2nd(sibling)\n(Sibling/Half-Sibling)',
    'G2b_GPGC':'2nd(GP-GC)\n(Grandparent-Grandchild)',
    'G3_3rd':'3rd\n(Uncle-Nephew/Great-Grandparent)',
    'G4_4th':'4th\n(Cousin/Grand-Uncle-Nephew)',
    'G5_5th':'5th\n(Cousin-Once-Removed)',
    'G6_6th':'6th\n(Second-Cousin)',
}
GROUP_TO_DEGREE = {
    'G0_Unrelated':0,'G1_1st':1,'G2a_Sib':2,'G2b_GPGC':2,
    'G3_3rd':3,'G4_4th':4,'G5_5th':5,'G6_6th':6
}
MARKER_COLORS = {
    'NFS_6K':'#BCD7EA','NFS_12K':'#84B9DA',
    'NFS_24K':'#4994C6','NFS_36K':'#2B6488',
    'Kintelligence':'#42B874','QIAseq':'#D5B32B'
}
GROUP_COLORS = {
    'G0_Unrelated':'#97A3A4','G1_1st':'#A8473E',
    'G2a_Sib':'#D25D51','G2b_GPGC':'#D25D51',
    'G3_3rd':'#CE803B','G4_4th':'#D5B32B',
    'G5_5th':'#42B874','G6_6th':'#4994C6'
}
METRIC_COLORS = {'IBS':'#4994C6','IBD':'#D25D51','Kinship':'#42B874'}


METRIC_DISPLAY = {'IBS': 'IBS', 'IBD': 'IBD', 'Kinship': 'KC'}
METRIC_FILEKEY = {'IBS': 'IBS', 'IBD': 'IBD', 'Kinship': 'KC'}

MARKER_ALIASES = {
    'Kintellignece': 'Kintelligence',
    'Qiaseq': 'QIAseq',
    'QIASeq': 'QIAseq',
    'Qiagen': 'QIAseq',
    'NFSKIN_36K': 'NFS_36K',
    'NFSKIN_24K': 'NFS_24K',
    'NFSKIN_12K': 'NFS_12K',
    'NFSKIN_6K': 'NFS_6K',
}
PLOT_MARKER_ORDER_FULL = ['NFS_36K', 'NFS_24K', 'NFS_12K', 'NFS_6K', 'Kintelligence', 'QIAseq']
PLOT_MARKER_ORDER_NFS_ONLY = ['NFS_36K', 'NFS_24K', 'NFS_12K', 'NFS_6K']
PAIRWISE_MARKER_SETS = [('NFS_12K', 'Kintelligence'), ('NFS_6K', 'QIAseq')]

MARKER_DISPLAY = {
    'NFS_36K': 'NFSKIN_36K',
    'NFS_24K': 'NFSKIN_24K',
    'NFS_12K': 'NFSKIN_12K',
    'NFS_6K': 'NFSKIN_6K',
    'Kintelligence': 'Kintelligence',
    'QIAseq': 'QIAseq',
}
MARKER_FILEKEY = {
    'NFS_36K': 'NFS_36K',
    'NFS_24K': 'NFS_24K',
    'NFS_12K': 'NFS_12K',
    'NFS_6K': 'NFS_6K',
    'Kintelligence': 'Kintelligence',
    'QIAseq': 'QIAseq',
}

def is_nocancer(ms):
    return isinstance(ms, str) and 'nocancer' in ms.lower()

def metric_display(metric):
    return METRIC_DISPLAY.get(metric, metric)

def metric_filekey(metric):
    return METRIC_FILEKEY.get(metric, metric)

def normalize_marker_name(ms):
    if pd.isna(ms):
        return ms
    return MARKER_ALIASES.get(ms, ms)

def normalize_marker_names(df):
    df = df.copy()
    if 'Marker_Set' in df.columns:
        df['Marker_Set'] = df['Marker_Set'].map(normalize_marker_name)
    return df

def marker_display(ms):
    return MARKER_DISPLAY.get(ms, ms)

def marker_filekey(ms):
    return MARKER_FILEKEY.get(ms, str(ms).replace(' ', '_'))

def variant_label(variant):
    return 'Full' if variant == 'full' else 'NFS-only'

def variant_filekey(variant):
    return 'full' if variant == 'full' else 'nfs_only'

def ordered_marker_list(marker_list):
    normalized = [normalize_marker_name(m) for m in marker_list if not is_nocancer(m)]
    ordered = [m for m in PLOT_MARKER_ORDER_FULL if m in normalized]
    extras = sorted([m for m in normalized if m not in ordered])
    return ordered + extras

def ordered_plot_markers(marker_list, variant='full'):
    normalized = ordered_marker_list(marker_list)
    base = PLOT_MARKER_ORDER_FULL if variant == 'full' else PLOT_MARKER_ORDER_NFS_ONLY
    return [m for m in base if m in normalized]

def filter_comparison_markers(marker_list):
    return ordered_plot_markers(marker_list, variant='full')

def apply_marker_order(df, markers, marker_col='Marker_Set'):
    out = df[df[marker_col].isin(markers)].copy()
    out[marker_col] = pd.Categorical(out[marker_col], categories=markers, ordered=True)
    return out.sort_values(marker_col)

def pairwise_label(marker_pair):
    return f"{marker_display(marker_pair[0])} vs {marker_display(marker_pair[1])}"

def _gd(g): return GROUP_DISPLAY.get(g,g)
def _gs(g): return GROUP_DISPLAY_SHORT.get(g,g)
def _gdr(g): return GROUP_DISPLAY_WITH_RELATIONSHIP.get(g,g)

# ============================================================
# 0. Data Prep
# ============================================================
def fix_degree_labels(df):
    df = df.copy()
    df['Relationship'] = df['Relationship'].replace({
        'Spouse': 'Unrelated',
        'In-Law': 'Unrelated',
        'InLaw': 'Unrelated',
        'Inlaw': 'Unrelated',
        'Spouse/InLaw': 'Unrelated',
        'Between-Fam': 'Unrelated',
    })
    df.loc[df['Relationship'].str.contains('inlaw|in-law|spouse|between', case=False, na=False), 'Relationship'] = 'Unrelated'
    corrections = 0
    for rel, cdeg in RELATIONSHIP_TO_DEGREE.items():
        mask = (df['Relationship']==rel) & (df['Degree']!=cdeg)
        n = mask.sum()
        if n>0:
            print(f"    FIX: {rel} degree {df.loc[mask,'Degree'].unique()} -> {cdeg} ({n} pairs)")
            df.loc[mask,'Degree'] = cdeg; corrections += n
    print(f"    Corrections: {corrections}" if corrections else "    No corrections needed.")
    df['Group'] = df['Relationship'].map(RELATIONSHIP_TO_GROUP)
    return df

def filter_real(df):
    return df[~(df['Sample1'].str.contains('GP|GM',na=False)|
                df['Sample2'].str.contains('GP|GM',na=False))].copy()

# ============================================================
# 1. ROC-based Threshold Classifier (Youden's J)
# ============================================================
def compute_group_stats(df_marker, metric):
    stats = {}
    for grp in df_marker['Group'].dropna().unique():
        v = df_marker[df_marker['Group']==grp][metric].dropna()
        if len(v)==0: continue
        stats[grp] = dict(n=len(v),mean=v.mean(),median=v.median(),std=v.std(),
                          min=v.min(),max=v.max(),
                          q10=v.quantile(.10),q25=v.quantile(.25),
                          q75=v.quantile(.75),q90=v.quantile(.90))
    return stats


def compute_roc_threshold(valsHi, valsLo):
    """Compute ROC optimal threshold (Youden's J) between two groups.

    valsHi: values of the group with higher metric values (positive class)
    valsLo: values of the group with lower metric values (negative class)

    Returns: (optimal_threshold, auc, midpoint_threshold)
    """
    scores = np.concatenate([valsHi, valsLo])
    # positive=1 for Hi group, negative=0 for Lo group
    labels = np.concatenate([np.ones(len(valsHi)), np.zeros(len(valsLo))])

    # Remove NaN
    valid = ~np.isnan(scores)
    scores, labels = scores[valid], labels[valid]

    if len(np.unique(labels)) < 2 or len(labels) == 0:
        midpoint = (np.median(valsHi) + np.median(valsLo)) / 2
        return midpoint, None, midpoint

    midpoint = (np.median(valsHi) + np.median(valsLo)) / 2

    try:
        auc = roc_auc_score(labels, scores)
        fpr, tpr, thresholds = roc_curve(labels, scores)
        # Youden's J = tpr - fpr, maximize
        j_scores = tpr - fpr
        opt_idx = np.argmax(j_scores)
        opt_threshold = thresholds[opt_idx]
        return opt_threshold, auc, midpoint
    except Exception:
        return midpoint, None, midpoint


def build_boundaries_roc(df_marker, metric, gstats):
    """Build ROC-based boundaries between adjacent groups.

    For each pair of adjacent groups (sorted by median), compute ROC curve
    and find Youden's J optimal threshold. Chain all thresholds.
    """
    medians = {g: s['median'] for g, s in gstats.items()}
    # Sort groups by median value (descending)
    ordered = sorted(medians, key=lambda g: medians[g], reverse=True)

    pair_thresholds = {}
    roc_details = {}
    for i in range(len(ordered) - 1):
        gHi, gLo = ordered[i], ordered[i + 1]
        vHi = df_marker[df_marker['Group']==gHi][metric].dropna().values
        vLo = df_marker[df_marker['Group']==gLo][metric].dropna().values

        if len(vHi) == 0 or len(vLo) == 0:
            th = (medians[gHi] + medians[gLo]) / 2
            pair_thresholds[(gHi, gLo)] = th
            roc_details[(gHi, gLo)] = dict(threshold=th, auc=None, midpoint=th,
                                           method='midpoint(fallback)')
            continue

        opt_th, auc, mid_th = compute_roc_threshold(vHi, vLo)
        pair_thresholds[(gHi, gLo)] = opt_th
        roc_details[(gHi, gLo)] = dict(threshold=opt_th, auc=auc, midpoint=mid_th,
                                       method='ROC(Youden)')

        auc_str = f"{auc:.3f}" if auc is not None else "N/A"
        print(f"      {_gs(gHi):>8} vs {_gs(gLo):<8}: "
              f"ROC_th={opt_th:.5f} (AUC={auc_str}) | midpoint={mid_th:.5f}")

    # Convert pair thresholds to per-group bounds
    bounds = {}
    for i, grp in enumerate(ordered):
        hi = float('inf') if i == 0 else pair_thresholds[(ordered[i-1], grp)]
        lo = float('-inf') if i == len(ordered)-1 else pair_thresholds[(grp, ordered[i+1])]
        bounds[grp] = (lo, hi)

    return bounds, roc_details


def classify_value(val, bounds):
    if pd.isna(val): return None
    for grp,(lo,hi) in bounds.items():
        if lo <= val < hi: return grp
    best,bestd = list(bounds.keys())[0], float('inf')
    for grp,(lo,hi) in bounds.items():
        mid = lo if hi==float('inf') else hi if lo==float('-inf') else (lo+hi)/2
        d = abs(val-mid)
        if d<bestd: bestd,best = d,grp
    return best

# ============================================================
# 2. Run
# ============================================================
def run_classifier(all_df, marker_list):
    all_results = []; cinfo = {}
    for ms in marker_list:
        print(f"\n  [{ms}]")
        df = filter_real(all_df[all_df['Marker_Set']==ms])
        cinfo[ms] = {}
        for metric in METRICS:
            if metric not in df.columns: continue
            gstats = compute_group_stats(df, metric)
            if not gstats: continue
            print(f"    {metric}:")
            bounds, roc_details = build_boundaries_roc(df, metric, gstats)
            cinfo[ms][metric] = dict(stats=gstats, boundaries=bounds, roc_details=roc_details)
            col_pred = f'Pred_{metric}'
            df[col_pred] = df[metric].apply(lambda v: classify_value(v, bounds))
            df[f'PredDeg_{metric}'] = df[col_pred].map(GROUP_TO_DEGREE)
            df[f'GroupCorrect_{metric}'] = df['Group']==df[col_pred]
            df[f'DegCorrect_{metric}'] = df['Degree']==df[f'PredDeg_{metric}']
            df[f'DegWithin1_{metric}'] = (df['Degree']-df[f'PredDeg_{metric}']).abs()<=1
            rel=df[df['Degree']>0]
            ga=rel[f'GroupCorrect_{metric}'].mean()*100 if len(rel) else 0
            da=rel[f'DegCorrect_{metric}'].mean()*100 if len(rel) else 0
            print(f"      => GroupAcc={ga:.1f}% DegreeAcc={da:.1f}%")
        all_results.append(df)
    return pd.concat(all_results, ignore_index=True), cinfo

# ============================================================
# 3. Evaluation
# ============================================================
def compute_group_accuracy(rdf,ml,metric):
    rows=[]
    pc,gc,dc,w1 = f'Pred_{metric}',f'GroupCorrect_{metric}',f'DegCorrect_{metric}',f'DegWithin1_{metric}'
    for ms in ml:
        df=rdf[rdf['Marker_Set']==ms].dropna(subset=[pc])
        for grp in [g for g in GROUP_ORDER if g in df['Group'].unique()]:
            s=df[df['Group']==grp]; n=len(s)
            rows.append(dict(Marker_Set=ms,Group=grp,Label=_gd(grp),
                Degree=GROUP_TO_DEGREE.get(grp,-1),N=n,Correct=int(s[gc].sum()),
                GroupAcc=s[gc].sum()/n*100 if n else 0,
                DegreeAcc=s[dc].sum()/n*100 if n else 0,
                Within1=s[w1].sum()/n*100 if n else 0))
    return pd.DataFrame(rows)

def compute_rel_accuracy(rdf,ml,metric):
    rows=[]
    pc,gc,dc = f'Pred_{metric}',f'GroupCorrect_{metric}',f'DegCorrect_{metric}'
    for ms in ml:
        df=rdf[rdf['Marker_Set']==ms].dropna(subset=[pc])
        for grp in [g for g in GROUP_ORDER if g in df['Group'].unique()]:
            s=df[df['Group']==grp]; n=len(s)
            pm=s[pc].mode()
            rows.append(dict(Marker_Set=ms,Relationship=_gd(grp),
                True_Degree=GROUP_TO_DEGREE.get(grp,-1),True_Group=grp,N=n,
                GroupAcc=s[gc].sum()/n*100 if n else 0,
                DegreeAcc=s[dc].sum()/n*100 if n else 0,
                MostPredGroup=pm.iloc[0] if len(pm) else '?'))
    return pd.DataFrame(rows)

def find_misclassified(rdf,ml,metric):
    rows=[]
    pc,gc = f'Pred_{metric}',f'GroupCorrect_{metric}'
    for ms in ml:
        df=rdf[rdf['Marker_Set']==ms].dropna(subset=[pc])
        mc=df[(df['Degree']>0)&(~df[gc])]
        for _,r in mc.iterrows():
            rows.append(dict(Marker_Set=ms,Family=r.get('Family1','?'),
                Sample1=r['Sample1'],Sample2=r['Sample2'],
                Member1=r.get('Member1',''),Member2=r.get('Member2',''),
                True_Relationship=r['Relationship'],
                True_Group=r['Group'],True_Degree=int(r['Degree']),
                Pred_Group=r[pc],
                Pred_Degree=int(GROUP_TO_DEGREE.get(r[pc],-1)),
                Metric_Value=round(r[metric],6) if pd.notna(r.get(metric)) else None))
    return pd.DataFrame(rows)

# ============================================================
# 4. Threshold Export
# ============================================================
def export_thresholds(cinfo, outdir):
    tdir=outdir/"thresholds"; tdir.mkdir(parents=True,exist_ok=True)
    for ms,minfo in cinfo.items():
        with open(tdir/f"thresholds_{ms}.txt",'w') as f:
            f.write(f"{'='*90}\nCLASSIFICATION THRESHOLDS: {marker_display(ms)}\n{'='*90}\n\n")
            f.write("Method: ROC-based (Youden's J index) per adjacent group pair\n")
            f.write("  threshold = argmax(TPR - FPR) on ROC curve\n")
            f.write("2nd degree split: Sibling vs GP-GC\n\n")
            for metric in METRICS:
                if metric not in minfo: continue
                bd=minfo[metric]['boundaries']; gs=minfo[metric]['stats']
                rd=minfo[metric]['roc_details']
                f.write(f"{'~'*90}\n  METRIC: {metric_display(metric)}\n{'~'*90}\n\n")

                # ROC details per boundary
                f.write(f"  ROC BOUNDARY DETAILS\n")
                f.write(f"  {'Pair':<30} {'Method':<18} {'ROC_Th':>10} {'AUC':>8} {'Midpoint':>10}\n")
                f.write(f"  "+"-"*80+"\n")
                for (gHi,gLo),det in rd.items():
                    pair_name = f"{_gs(gHi)} vs {_gs(gLo)}"
                    auc_s = f"{det['auc']:.3f}" if det['auc'] is not None else "N/A"
                    f.write(f"  {pair_name:<30} {det['method']:<18} "
                            f"{det['threshold']:>10.6f} {auc_s:>8} {det['midpoint']:>10.6f}\n")

                f.write(f"\n  CLASSIFICATION BOUNDARIES\n")
                f.write(f"  {'Group':<22} {'Lower':>12} {'Upper':>12}  |  {'Median':>10} {'Mean':>10} {'Std':>8} {'N':>5}\n")
                f.write(f"  "+"-"*85+"\n")
                for grp in GROUP_ORDER:
                    if grp not in bd: continue
                    lo,hi=bd[grp]; s=gs.get(grp,{})
                    lo_s=f"{lo:.6f}" if lo!=float('-inf') else "      -inf"
                    hi_s=f"{hi:.6f}" if hi!=float('inf') else "      +inf"
                    f.write(f"  {_gd(grp):<22} {lo_s:>12} {hi_s:>12}  |  "
                            f"{s.get('median',0):>10.6f} {s.get('mean',0):>10.6f} "
                            f"{s.get('std',0):>8.6f} {s.get('n',0):>5}\n")
                f.write(f"\n  STATISTICS\n")
                f.write(f"  {'Group':<22} {'N':>5} {'Min':>9} {'Q10':>9} {'Q25':>9} {'Median':>9} {'Q75':>9} {'Q90':>9} {'Max':>9}\n")
                f.write(f"  "+"-"*90+"\n")
                for grp in GROUP_ORDER:
                    if grp not in gs: continue
                    s=gs[grp]
                    f.write(f"  {_gd(grp):<22} {s['n']:>5} {s['min']:>9.5f} {s['q10']:>9.5f} "
                            f"{s['q25']:>9.5f} {s['median']:>9.5f} {s['q75']:>9.5f} {s['q90']:>9.5f} {s['max']:>9.5f}\n")
                f.write("\n")
            f.write(f"{'='*90}\n")
        print(f"    Saved: thresholds_{ms}.txt")
        csv_rows=[]
        for metric in METRICS:
            if metric not in minfo: continue
            bd=minfo[metric]['boundaries']; gs=minfo[metric]['stats']
            rd=minfo[metric]['roc_details']
            for grp in GROUP_ORDER:
                if grp not in bd: continue
                lo,hi=bd[grp]; s=gs.get(grp,{})
                # find AUC for boundaries involving this group
                auc_lo = next((d['auc'] for (gH,gL),d in rd.items() if gL==grp), None)
                auc_hi = next((d['auc'] for (gH,gL),d in rd.items() if gH==grp), None)
                csv_rows.append(dict(Metric=metric_display(metric),RawMetric=metric,Group=grp,GroupLabel=_gs(grp),
                    Degree=GROUP_TO_DEGREE.get(grp,-1),
                    Lower=lo if lo!=float('-inf') else None,
                    Upper=hi if hi!=float('inf') else None,
                    Median=s.get('median'),Mean=s.get('mean'),
                    Std=s.get('std'),N=s.get('n'),Q25=s.get('q25'),Q75=s.get('q75'),
                    AUC_Lower=auc_lo, AUC_Upper=auc_hi))
        pd.DataFrame(csv_rows).to_csv(tdir/f"thresholds_{ms}.csv",index=False)


# ============================================================
# 5. Plots
# ============================================================
def build_accuracy_summary(rdf, markers, metric):
    pc, gc = f'Pred_{metric}', f'GroupCorrect_{metric}'
    rows = []
    for ms in markers:
        df = rdf[rdf['Marker_Set'] == ms].dropna(subset=[pc])
        n = len(df)
        rel = df[df['Degree'] > 0]
        rows.append(dict(
            Marker_Set=ms,
            All=df[gc].mean() * 100 if n else 0,
            Related=rel[gc].mean() * 100 if len(rel) else 0,
        ))
    return pd.DataFrame(rows)

def plot_accuracy_overall(rdf, markers, metric, fdir, variant):
    if not markers:
        return
    summary = build_accuracy_summary(rdf, markers, metric)
    if len(summary) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, col, subtitle in [
        (axes[0], 'All', 'All pairs'),
        (axes[1], 'Related', 'Related only'),
    ]:
        vals = [summary.loc[summary['Marker_Set'] == ms, col].iloc[0] for ms in markers]
        colors = [MARKER_COLORS.get(ms, '#97A3A4') for ms in markers]
        bars = ax.bar(range(len(markers)), vals, color=colors, edgecolor='white')
        ax.set_xticks(range(len(markers)))
        ax.set_xticklabels([marker_display(m) for m in markers], rotation=45, ha='right', fontsize=10)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'[{metric_display(metric)}] Overall accuracy ({variant_label(variant)}) - {subtitle}',
                     fontweight='bold')
        ax.set_ylim(0, 105)
        ax.grid(axis='y', alpha=.3)
        for b in bars:
            ax.annotate(f'{b.get_height():.1f}%',
                        xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    out_name = f"accuracy_overall_{metric_filekey(metric)}_{variant_filekey(variant)}.png"
    plt.savefig(fdir / out_name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {out_name}")

def plot_accuracy_heatmap_group(gadf, markers, metric, path, variant):
    if not markers:
        return
    gadf = apply_marker_order(gadf, markers)
    if len(gadf) == 0:
        return
    groups = [g for g in GROUP_ORDER if g in gadf['Group'].values]
    pivot = gadf.pivot_table(index='Marker_Set', columns='Group', values='GroupAcc', aggfunc='first')
    pivot = pivot.reindex(index=markers, columns=groups)
    fig, ax = plt.subplots(figsize=(max(12, len(groups) * 1.8), max(5, len(markers) * 0.9)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.1f',
        cmap='mako',
        vmin=0,
        vmax=100,
        linewidths=.5,
        linecolor='white',
        ax=ax,
        cbar_kws={'label': 'Accuracy (%)', 'shrink': .8},
        annot_kws={'fontsize': 10, 'fontweight': 'bold'},
    )
    ax.set_xticklabels([_gd(g) for g in groups], rotation=25, ha='right', fontsize=9)
    ax.set_yticklabels([marker_display(m) for m in markers], rotation=0, fontsize=11)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Marker set')
    ax.set_title(f'[{metric_display(metric)}] Group accuracy heatmap ({variant_label(variant)})',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path.name}")

def plot_accuracy_by_relationship(radf, markers, metric, path, variant):
    if not markers:
        return
    df = radf[radf['Marker_Set'].isin(markers)].copy()
    if len(df) == 0:
        return
    groups = [g for g in GROUP_ORDER if g in df['True_Group'].values]
    fig, ax = plt.subplots(figsize=(max(14, len(groups) * 1.8), 7))
    nm = len(markers)
    bw = 0.8 / nm
    x = np.arange(len(groups))
    for i, ms in enumerate(markers):
        sub = df[df['Marker_Set'] == ms].set_index('True_Group')
        vals = [sub.loc[g, 'GroupAcc'] if g in sub.index else 0 for g in groups]
        bars = ax.bar(x + (i - nm / 2 + .5) * bw, vals, bw, label=marker_display(ms),
                      color=MARKER_COLORS.get(ms, '#97A3A4'), edgecolor='white', linewidth=.3)
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.annotate(f'{h:.0f}',
                            xy=(b.get_x() + b.get_width() / 2, h),
                            ha='center', va='bottom', fontsize=6, fontweight='bold', rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([_gd(g) for g in groups], rotation=25, ha='right', fontsize=9)
    ax.set_xlabel('Degree')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 120)
    ax.set_title(f'[{metric_display(metric)}] Empirical relationship accuracy ({variant_label(variant)})',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(axis='y', alpha=.3)
    ax.axhline(100, color='gray', ls='--', alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path.name}")

def plot_confusion_matrices(rdf, markers, metric, outdir):
    pc = f'Pred_{metric}'
    for ms in markers:
        df = rdf[rdf['Marker_Set'] == ms].dropna(subset=[pc])
        if len(df) == 0:
            continue
        yt = df['Group'].astype(str)
        yp = df[pc].astype(str)
        labels = [g for g in GROUP_ORDER if g in set(yt) | set(yp)]
        cm = confusion_matrix(yt, yp, labels=labels)
        cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
        tl = [_gd(l) for l in labels]

        count_name = f"confusion_counts_{metric_filekey(metric)}_{marker_filekey(ms)}.png"
        fig, ax = plt.subplots(figsize=(11, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='mako', ax=ax,
                    xticklabels=tl, yticklabels=tl, linewidths=.5, linecolor='white',
                    annot_kws={'fontsize': 18, 'fontweight': 'normal'},
                    cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted', fontsize=20, fontweight='normal')
        ax.set_ylabel('True', fontsize=20, fontweight='normal')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=18, fontweight='normal')
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=18, fontweight='normal')
        plt.tight_layout()
        plt.savefig(outdir / count_name, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {count_name}")

        norm_name = f"confusion_row_normalized_{metric_filekey(metric)}_{marker_filekey(ms)}.png"
        fig, ax = plt.subplots(figsize=(11, 10))
        sns.heatmap(cmn, annot=True, fmt='.1f', cmap='mako', vmin=0, vmax=100, ax=ax,
                    xticklabels=tl, yticklabels=tl, linewidths=.5, linecolor='white',
                    annot_kws={'fontsize': 18, 'fontweight': 'normal'},
                    cbar_kws={'label': 'Row-normalized (%)'})
        ax.set_xlabel('Predicted', fontsize=20, fontweight='normal')
        ax.set_ylabel('True', fontsize=20, fontweight='normal')
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=18, fontweight='normal')
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=18, fontweight='normal')
        plt.tight_layout()
        plt.savefig(outdir / norm_name, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {norm_name}")

def plot_confusion_triplets_row_normalized(rdf, markers, outdir):
    """Save one row-normalized confusion-matrix triplet per marker.

    The triplet places IBS, IBD, and KC side-by-side with a single shared
    percentage colorbar on the far right, while keeping the existing
    per-metric confusion plots unchanged.
    """
    metrics = ['IBS', 'IBD', 'Kinship']
    for ms in markers:
        marker_df = rdf[rdf['Marker_Set'] == ms]
        if len(marker_df) == 0:
            continue

        observed = set(marker_df['Group'].dropna().astype(str))
        predicted = set()
        for metric in metrics:
            pc = f'Pred_{metric}'
            if pc in marker_df.columns:
                predicted.update(marker_df[pc].dropna().astype(str))
        labels = [g for g in GROUP_ORDER if g in observed | predicted]
        if not labels:
            continue
        tl = [_gd(l) for l in labels]
        ytl = tl

        fig, axes = plt.subplots(1, 3, figsize=(30, 9), sharey=False)
        cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.66])
        rendered = False
        for idx, (ax, metric) in enumerate(zip(axes, metrics)):
            pc = f'Pred_{metric}'
            df = marker_df.dropna(subset=[pc]) if pc in marker_df.columns else marker_df.iloc[0:0]
            if len(df) == 0:
                cmn = np.zeros((len(labels), len(labels)))
                annot = np.full((len(labels), len(labels)), '', dtype=object)
            else:
                yt = df['Group'].astype(str)
                yp = df[pc].astype(str)
                cm = confusion_matrix(yt, yp, labels=labels)
                cmn = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100
                annot = np.vectorize(lambda v: f'{v:.1f}')(cmn)
                rendered = True

            sns.heatmap(cmn, annot=annot, fmt='', cmap='mako', vmin=0, vmax=100, ax=ax,
                        xticklabels=tl, yticklabels=ytl if idx == 0 else False,
                        linewidths=.5, linecolor='white',
                        annot_kws={'fontsize': 14, 'fontweight': 'normal'},
                        cbar=(idx == 2), cbar_ax=cbar_ax if idx == 2 else None,
                        cbar_kws={'label': 'Row-normalized (%)'} if idx == 2 else None)
            ax.set_xlabel('Predicted', fontsize=18, fontweight='normal')
            ax.set_ylabel('True', fontsize=18, fontweight='normal')
            ax.text(0.5, 1.02, metric_display(metric), transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=20, fontweight='normal')
            plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=14, fontweight='normal')
            plt.setp(ax.get_yticklabels(), rotation=0, fontsize=14, fontweight='normal')

        if not rendered:
            plt.close()
            continue
        fig.subplots_adjust(left=0.12, right=0.90, bottom=0.16, top=0.88, wspace=0.08)
        triplet_name = f"confusion_row_normalized_triplet_{marker_filekey(ms)}.png"
        plt.savefig(outdir / triplet_name, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"    Saved: {triplet_name}")

def plot_misclassification_summary(mcdf, markers, metric, path, variant):
    if len(mcdf) == 0 or not markers:
        print("    No misclassifications.")
        return
    mcdf_f = mcdf[mcdf['Marker_Set'].isin(markers)].copy()
    if len(mcdf_f) == 0:
        print("    No misclassifications.")
        return
    pivot = mcdf_f.groupby(['Marker_Set', 'True_Group']).size().reset_index(name='Errors')
    groups = [g for g in GROUP_ORDER if g in pivot['True_Group'].values]
    fig, ax = plt.subplots(figsize=(max(12, len(groups) * 2), 6))
    nm = len(markers)
    bw = 0.8 / nm
    x = np.arange(len(groups))
    for i, ms in enumerate(markers):
        sub = pivot[pivot['Marker_Set'] == ms].set_index('True_Group')
        vals = [sub.loc[g, 'Errors'] if g in sub.index else 0 for g in groups]
        bars = ax.bar(x + (i - nm / 2 + .5) * bw, vals, bw, label=marker_display(ms),
                      color=MARKER_COLORS.get(ms, '#97A3A4'), edgecolor='white', linewidth=.3)
        for b in bars:
            h = b.get_height()
            if h > 0:
                ax.annotate(f'{int(h)}',
                            xy=(b.get_x() + b.get_width() / 2, h),
                            ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([_gd(g) for g in groups], fontsize=9, rotation=15, ha='right')
    ax.set_ylabel('# misclassified pairs')
    ax.set_title(f'[{metric_display(metric)}] Misclassification counts ({variant_label(variant)})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path.name}")

def plot_forensic_scenarios(rdf, markers, metric, path, variant):
    if not markers:
        return
    gc, w1, pc = f'GroupCorrect_{metric}', f'DegWithin1_{metric}', f'Pred_{metric}'
    fig, axes = plt.subplots(2, 1, figsize=(16, 12))
    ax = axes[0]
    all_groups = [g for g in GROUP_ORDER if g != 'G0_Unrelated']
    for ms in markers:
        df = rdf[(rdf['Marker_Set'] == ms) & (rdf['Degree'] > 0)].dropna(subset=[pc])
        accs = [df[df['Group'] == g][gc].mean() * 100 if len(df[df['Group'] == g]) else 0 for g in all_groups]
        ax.plot(range(len(all_groups)), accs, 'o-', label=marker_display(ms),
                color=MARKER_COLORS.get(ms, 'gray'), lw=2, ms=8)
    ax.set_xticks(range(len(all_groups)))
    ax.set_xticklabels([_gd(g) for g in all_groups], rotation=15, ha='right', fontsize=9)
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.grid(alpha=.3)
    ax.set_title(f'[{metric_display(metric)}] Group accuracy by kinship distance ({variant_label(variant)})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)

    ax = axes[1]
    distant = []
    for ms in markers:
        df = rdf[(rdf['Marker_Set'] == ms) & (rdf['Degree'].isin([5, 6]))].dropna(subset=[pc])
        distant.append(dict(Marker_Set=ms,
                            Exact=df[gc].mean() * 100 if len(df) else 0,
                            Within1=df[w1].mean() * 100 if len(df) else 0))
    distant = pd.DataFrame(distant)
    x = np.arange(len(distant))
    w = 0.35
    b1 = ax.bar(x - w / 2, distant['Exact'], w, label='Exact',
                color='#4994C6', edgecolor='white')
    b2 = ax.bar(x + w / 2, distant['Within1'], w, label='Within ±1',
                color='#42B874', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([marker_display(m) for m in distant['Marker_Set']], rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=.3)
    ax.set_title(f'[{metric_display(metric)}] 5th-6th degree focus ({variant_label(variant)})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    for b in list(b1) + list(b2):
        ax.annotate(f'{b.get_height():.0f}%',
                    xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path.name}")

def plot_thresholds(cinfo, ms, metric, path):
    if metric not in cinfo.get(ms, {}):
        return
    bd = cinfo[ms][metric]['boundaries']
    gs = cinfo[ms][metric]['stats']
    grps = [g for g in THRESHOLD_PLOT_ORDER if g in gs]
    fig, ax = plt.subplots(figsize=(10, max(5, len(grps) * 0.8)))
    for i, grp in enumerate(grps):
        s = gs[grp]
        c = GROUP_COLORS.get(grp, '#97A3A4')
        ax.barh(i, s['q75'] - s['q25'], left=s['q25'], height=.6,
                color=c, alpha=.6, edgecolor='black', lw=.5)
        ax.plot(s['median'], i, 'k|', ms=15, mew=2)
        ax.plot([s['min'], s['max']], [i, i], 'k-', lw=.5, alpha=.5)
        ax.annotate(f'{s["median"]:.4f}',
                    xy=(s['median'], i + 0.35),
                    ha='center', va='bottom', fontsize=7, color='black', fontweight='bold')
    drawn = set()
    for _, (lo, hi) in bd.items():
        for th in [lo, hi]:
            if th not in (float('inf'), float('-inf')):
                th_r = round(th, 8)
                if th_r not in drawn:
                    ax.axvline(th, color='red', ls='--', alpha=.6, lw=1.2)
                    ax.annotate(f'{th:.4f}', xy=(th, -0.7), ha='center', va='top',
                                fontsize=7, color='red', fontweight='bold', rotation=45)
                    drawn.add(th_r)
    ax.set_yticks(range(len(grps)))
    ax.set_yticklabels([_gd(g) for g in grps], fontsize=9)
    ax.set_xlabel(metric_display(metric), fontsize=12)
    ax.set_title(f'[{metric_display(metric)}] Threshold ranges - {marker_display(ms)}',
                 fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=.3)
    ax.set_ylim(-1.5, len(grps) - 0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path.name}")

def plot_metric_comparison(rdf, markers, path, title_suffix=''):
    if not markers:
        return
    rows = []
    for ms in markers:
        df = rdf[rdf['Marker_Set'] == ms]
        for metric in METRICS:
            gc = f'GroupCorrect_{metric}'
            if gc not in df.columns:
                continue
            valid = df.dropna(subset=[f'Pred_{metric}'])
            rel = valid[valid['Degree'] > 0]
            rows.append(dict(
                Marker_Set=ms,
                Metric=metric,
                All=valid[gc].mean() * 100 if len(valid) else 0,
                Related=rel[gc].mean() * 100 if len(rel) else 0,
            ))
    mdf = pd.DataFrame(rows)
    if len(mdf) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
    for ax, col, subtitle in zip(axes, ['All', 'Related'], ['All pairs', 'Related only']):
        x = np.arange(len(markers))
        w = 0.25
        for i, metric in enumerate(METRICS):
            vals = []
            for ms in markers:
                sub = mdf[(mdf['Marker_Set'] == ms) & (mdf['Metric'] == metric)]
                vals.append(sub[col].iloc[0] if len(sub) else 0)
            bars = ax.bar(x + (i - 1) * w, vals, w, label=metric_display(metric),
                          color=METRIC_COLORS[metric], edgecolor='white')
            for b in bars:
                ax.annotate(f'{b.get_height():.1f}',
                            xy=(b.get_x() + b.get_width() / 2, b.get_height() + 0.5),
                            ha='center', va='bottom', fontsize=7, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([marker_display(m) for m in markers], rotation=45, ha='right')
        ax.set_ylabel('Accuracy (%)')
        suffix = f' - {title_suffix}' if title_suffix else ''
        ax.set_title(f'Metric comparison ({subtitle}){suffix}', fontweight='bold')
        ax.set_ylim(0, 110)
        ax.grid(axis='y', alpha=.3)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.0), ncol=3,
                  frameon=True, borderaxespad=0.3)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path.name}")

def plot_pairwise_relationship_accuracy(all_radf, marker_pair, path):
    markers = [normalize_marker_name(m) for m in marker_pair]
    rows = []
    rel_order = GROUP_ORDER
    for metric in METRICS:
        radf = all_radf.get(metric)
        if radf is None or len(radf) == 0:
            continue
        sub = radf[radf['Marker_Set'].isin(markers)].copy()
        if len(sub) == 0:
            continue
        for ms in markers:
            marker_sub = sub[sub['Marker_Set'] == ms].set_index('True_Group')
            for rel in rel_order:
                if rel in marker_sub.index:
                    rows.append(dict(
                        Relationship=_gd(rel),
                        Column=f'{marker_display(ms)}\n{metric_display(metric)}',
                        GroupAcc=marker_sub.loc[rel, 'GroupAcc'],
                    ))
    if not rows:
        return
    col_order = [f'{marker_display(ms)}\n{metric_display(metric)}' for ms in markers for metric in METRICS]
    pivot = pd.DataFrame(rows).pivot_table(index='Relationship', columns='Column', values='GroupAcc', aggfunc='first')
    pivot = pivot.reindex(index=[_gd(r) for r in rel_order if _gd(r) in pivot.index], columns=col_order)
    fig, ax = plt.subplots(figsize=(max(10, len(col_order) * 1.5), max(6, len(pivot.index) * 0.8)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt='.1f',
        cmap='mako',
        vmin=0,
        vmax=100,
        linewidths=.5,
        linecolor='white',
        ax=ax,
        cbar_kws={'label': 'Accuracy (%)', 'shrink': .8},
        annot_kws={'fontsize': 9, 'fontweight': 'bold'},
    )
    ax.set_xlabel('Panel / metric')
    ax.set_ylabel('Degree')
    ax.set_title(f'Pairwise empirical relationship accuracy - {pairwise_label(markers)}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {path.name}")

# ============================================================
# 6. Master Tables & Report
# ============================================================
def generate_master_tables(rdf,mlist,tdir):
    for ms in mlist:
        df=rdf[rdf['Marker_Set']==ms].copy()
        cols=['Sample1','Sample2','Family1','Relationship','Degree','Group','IBS','IBD','Kinship']
        for m in METRICS: cols+=[f'Pred_{m}',f'PredDeg_{m}',f'GroupCorrect_{m}',f'DegCorrect_{m}']
        avail=[c for c in cols if c in df.columns]
        df[avail].sort_values(['Degree','Family1','Sample1','Sample2'],
            ascending=[False,True,True,True]).to_csv(tdir/f"master_table_{ms}.csv",index=False)
        print(f"    Saved: master_table_{ms}.csv")


def generate_report(rdf, all_gadf, all_radf, all_mcdf, cinfo, mlist, rpath):
    ordered_report_markers = ordered_marker_list(mlist)
    with open(rpath, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\nKINSHIP CLASSIFIER - EVALUATION REPORT (v8)\n" + "=" * 100 + "\n\n")
        f.write("METHOD\n" + "-" * 80 + "\n")
        f.write("  Per-metric classification (IBS, IBD, KC independently)\n")
        f.write("  ROC-based thresholds: Youden's J (argmax TPR-FPR) per adjacent group pair\n")
        f.write("  2nd degree split: Sibling vs GP-GC\n")
        f.write("  Grand-Uncle-Nephew = 4th degree\n\n")
        for metric in METRICS:
            metric_label = metric_display(metric)
            gadf = all_gadf[metric]
            radf = all_radf[metric]
            mcdf = all_mcdf[metric]
            f.write(f"\n{'#' * 100}\n  METRIC: {metric_label}\n{'#' * 100}\n\n")
            f.write(f"  1. SUMMARY\n  " + "-" * 70 + "\n\n")
            for ms in ordered_report_markers:
                gc, dc = f'GroupCorrect_{metric}', f'DegCorrect_{metric}'
                df = rdf[rdf['Marker_Set'] == ms].dropna(subset=[f'Pred_{metric}'])
                n = len(df)
                nc = int(df[gc].sum())
                ndc = int(df[dc].sum())
                rel = df[df['Degree'] > 0]
                nr = len(rel)
                nrc = int(rel[gc].sum()) if nr else 0
                nrdc = int(rel[dc].sum()) if nr else 0
                f.write(f"    [{marker_display(ms)}]\n")
                f.write(f"      All:     {n:>6} | GroupAcc: {nc / n * 100:>5.1f}% | DegreeAcc: {ndc / n * 100:>5.1f}%\n")
                if nr:
                    f.write(f"      Related: {nr:>6} | GroupAcc: {nrc / nr * 100:>5.1f}% | DegreeAcc: {nrdc / nr * 100:>5.1f}%\n")
                f.write("\n")

            f.write(f"\n  2. ROC THRESHOLDS\n  " + "-" * 70 + "\n")
            for ms in ordered_report_markers:
                if metric not in cinfo.get(ms, {}):
                    continue
                rd = cinfo[ms][metric]['roc_details']
                f.write(f"\n    [{marker_display(ms)}]\n")
                f.write(f"    {'Pair':<30} {'ROC_Th':>10} {'AUC':>8} {'Midpoint':>10}\n")
                f.write(f"    " + "-" * 60 + "\n")
                for (gHi, gLo), det in rd.items():
                    pair_name = f"{_gs(gHi)} vs {_gs(gLo)}"
                    auc_s = f"{det['auc']:.3f}" if det['auc'] is not None else "N/A"
                    f.write(f"    {pair_name:<30} {det['threshold']:>10.6f} {auc_s:>8} {det['midpoint']:>10.6f}\n")

            f.write(f"\n\n  3. PER-GROUP ACCURACY\n  " + "-" * 70 + "\n")
            for ms in ordered_report_markers:
                f.write(f"\n    [{marker_display(ms)}]\n    {'Group':<22} {'N':>5} {'GrpAcc':>7} {'DegAcc':>7} {'+/-1':>6}\n    " + "-" * 50 + "\n")
                for _, r in gadf[gadf['Marker_Set'] == ms].sort_values('Group').iterrows():
                    f.write(f"    {r['Label']:<22} {int(r['N']):>5} {r['GroupAcc']:>6.1f}% {r['DegreeAcc']:>6.1f}% {r['Within1']:>5.1f}%\n")

            f.write(f"\n\n  4. PER-RELATIONSHIP ACCURACY\n  " + "-" * 70 + "\n")
            for ms in ordered_report_markers:
                f.write(f"\n    [{marker_display(ms)}]\n    {'Relationship':<26} {'Group':>10} {'N':>5} {'GrpAcc':>7} {'DegAcc':>7} {'->':>10}\n    " + "-" * 70 + "\n")
                sub = radf[(radf['Marker_Set'] == ms) & (radf['True_Degree'] > 0)].sort_values('True_Group')
                for _, r in sub.iterrows():
                    f.write(f"    {r['Relationship']:<26} {_gs(r['True_Group']):>10} {int(r['N']):>5} "
                            f"{r['GroupAcc']:>6.1f}% {r['DegreeAcc']:>6.1f}% ->{_gs(r['MostPredGroup']):>9}\n")

            f.write(f"\n\n  5. MISCLASSIFIED (related)\n  " + "-" * 70 + "\n")
            if len(mcdf) == 0:
                f.write("    None!\n")
            else:
                for ms in ordered_report_markers:
                    sub = mcdf[mcdf['Marker_Set'] == ms]
                    if len(sub) == 0:
                        f.write(f"\n    [{marker_display(ms)}] None\n")
                        continue
                    f.write(f"\n    [{marker_display(ms)}] {len(sub)} misclassified\n")
                    f.write(f"    {'Fam':>4} {'S1':<16} {'S2':<16} {'Relation':<22} {'True':>10} {'Pred':>10} {metric_label:>10}\n    " + "-" * 95 + "\n")
                    for _, r in sub.sort_values(['True_Group', 'Family']).iterrows():
                        mv = f"{r['Metric_Value']:.5f}" if pd.notna(r.get('Metric_Value')) else 'N/A'
                        f.write(f"    {r['Family']:>4} {str(r['Sample1']):<16} {str(r['Sample2']):<16} "
                                f"{r['True_Relationship']:<22} {_gs(r['True_Group']):>10} -> {_gs(r['Pred_Group']):>8} {mv:>10}\n")
        f.write("\n\n" + "=" * 100 + "\nEND OF REPORT\n" + "=" * 100 + "\n")
    print(f"    Saved: {rpath.name}")


# ============================================================
# Main
# ============================================================

def default_eval_dir_for_analysis(analysis_dir):
    """Return the sibling Step 5 directory for a Step 3/4 analysis directory."""
    analysis_path = Path(analysis_dir)
    if analysis_path.name.startswith("06_"):
        return analysis_path.parent / DEFAULT_EVAL_SUBDIR
    return DEFAULT_EVAL_DIR


def resolve_combined_csv(args):
    """Resolve Step 5's combined result table from explicit or pipeline paths."""
    if args.combined_csv:
        candidates = [Path(args.combined_csv)]
    else:
        analysis_dir = Path(args.analysis_dir) if args.analysis_dir else DEFAULT_ANALYSIS_DIR
        eval_dir = Path(args.eval_dir) if args.eval_dir else default_eval_dir_for_analysis(analysis_dir)
        candidates = [
            eval_dir / COMBINED_CSV_NAME,
            DEFAULT_EVAL_DIR / COMBINED_CSV_NAME,
            analysis_dir / DEFAULT_EVAL_SUBDIR / COMBINED_CSV_NAME,
            analysis_dir / COMBINED_CSV_NAME,
            DEFAULT_WORK_DIR / COMBINED_CSV_NAME,
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = "\n  - ".join(str(c) for c in candidates)
    print(f"ERROR: {COMBINED_CSV_NAME} not found. Checked:\n  - {checked}")
    sys.exit(1)


def resolve_output_dir(args, cpath):
    """Resolve classifier output directory under Step 5 outputs by default."""
    if args.output_dir:
        return Path(args.output_dir)
    return cpath.parent / DEFAULT_CLASSIFIER_SUBDIR


def parse_args():
    parser = argparse.ArgumentParser(
        description='Step 6: classify Step 5 kinship evaluation results with per-metric ROC thresholds')
    parser.add_argument('--analysis-dir', '--results-dir', dest='analysis_dir', type=str,
                        default=str(DEFAULT_ANALYSIS_DIR),
                        help='Step 3/4 analysis directory; Step 5 defaults to sibling 07_evaluate_kinship (default: %(default)s)')
    parser.add_argument('--eval-dir', type=str, default=None,
                        help='Step 5 evaluation output directory containing all_results_combined.csv')
    parser.add_argument('--combined-csv', type=str,
                        help='Explicit Step 5 all_results_combined.csv path')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Classifier output directory (default: <eval-dir>/classifier)')
    return parser.parse_args()


def main():
    args = parse_args()
    cpath = resolve_combined_csv(args)

    print("=" * 70 + "\nSTEP 6: KINSHIP CLASSIFIER v8 (per-metric, ROC-based thresholds)\n" + "=" * 70)
    print(f"\n[1] Loading Step 5 combined results: {cpath}")
    all_df = pd.read_csv(cpath)
    all_df = normalize_marker_names(all_df)
    marker_list = ordered_marker_list(all_df['Marker_Set'].dropna().unique())
    ml = [m for m in marker_list if not is_nocancer(m)]
    ml_plot_full = ordered_plot_markers(ml, variant='full')
    ml_plot_nfs = ordered_plot_markers(ml, variant='nfs_only')
    print(f"    {len(all_df):,} rows, {len(marker_list)} markers ({len(ml)} non-nocancer)")
    print(f"    Full plot markers: {', '.join([marker_display(m) for m in ml_plot_full])}")
    print(f"    NFS-only markers: {', '.join([marker_display(m) for m in ml_plot_nfs])}")
    print(f"\n[1b] Fixing degrees & adding groups...")
    all_df = fix_degree_labels(all_df)

    outdir = resolve_output_dir(args, cpath)
    fdir, tdir = outdir / "figures", outdir / "tables"
    for d in [outdir, fdir, tdir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"\n[2] Classifying (per-metric, ROC Youden's J)...")
    rdf, cinfo = run_classifier(all_df, ml)

    print(f"\n[3] Evaluating...")
    all_gadf, all_radf, all_mcdf = {}, {}, {}
    for metric in METRICS:
        print(f"\n  --- {metric_display(metric)} ---")
        gadf = compute_group_accuracy(rdf, ml, metric)
        gadf.to_csv(tdir / f"accuracy_group_{metric_filekey(metric)}.csv", index=False)
        radf = compute_rel_accuracy(rdf, ml, metric)
        radf.to_csv(tdir / f"accuracy_rel_{metric_filekey(metric)}.csv", index=False)
        mcdf = find_misclassified(rdf, ml, metric)
        mcdf.to_csv(tdir / f"misclassified_{metric_filekey(metric)}.csv", index=False)
        print(f"    Misclassified (related): {len(mcdf)}")
        all_gadf[metric] = gadf
        all_radf[metric] = radf
        all_mcdf[metric] = mcdf

    print(f"\n[4] Thresholds...")
    export_thresholds(cinfo, outdir)
    print(f"\n[5] Master tables...")
    generate_master_tables(rdf, ml, tdir)

    print(f"\n[6] Figures...")
    single_marker_plots = ml_plot_full
    for metric in METRICS:
        metric_key = metric_filekey(metric)
        print(f"\n  --- {metric_display(metric)} ---")
        for variant, markers in [('full', ml_plot_full), ('nfs_only', ml_plot_nfs)]:
            if not markers:
                continue
            gadf_subset = all_gadf[metric][all_gadf[metric]['Marker_Set'].isin(markers)]
            radf_subset = all_radf[metric][all_radf[metric]['Marker_Set'].isin(markers)]
            plot_accuracy_overall(rdf, markers, metric, fdir, variant)
            plot_accuracy_heatmap_group(
                gadf_subset,
                markers,
                metric,
                fdir / f"group_accuracy_heatmap_{metric_key}_{variant_filekey(variant)}.png",
                variant,
            )
            plot_accuracy_by_relationship(
                radf_subset,
                markers,
                metric,
                fdir / f"relationship_accuracy_{metric_key}_{variant_filekey(variant)}.png",
                variant,
            )
            plot_misclassification_summary(
                all_mcdf[metric],
                markers,
                metric,
                fdir / f"misclassification_counts_{metric_key}_{variant_filekey(variant)}.png",
                variant,
            )
            plot_forensic_scenarios(
                rdf,
                markers,
                metric,
                fdir / f"extended_kinship_{metric_key}_{variant_filekey(variant)}.png",
                variant,
            )
        plot_confusion_matrices(rdf, single_marker_plots, metric, fdir)
        if metric == METRICS[-1]:
            plot_confusion_triplets_row_normalized(rdf, single_marker_plots, fdir)
        for ms in single_marker_plots:
            plot_thresholds(
                cinfo,
                ms,
                metric,
                fdir / f"thresholds_{metric_key}_{marker_filekey(ms)}.png",
            )

    print(f"\n  --- Metric comparison ---")
    plot_metric_comparison(rdf, ml_plot_full, fdir / "metric_comparison_full.png", title_suffix='Full')
    plot_metric_comparison(rdf, ml_plot_nfs, fdir / "metric_comparison_nfs_only.png", title_suffix='NFS-only')

    for left, right in PAIRWISE_MARKER_SETS:
        pair = [m for m in [left, right] if m in ml]
        if len(pair) != 2:
            continue
        pair_slug = f"{marker_filekey(pair[0])}_vs_{marker_filekey(pair[1])}"
        pair_title = pairwise_label(pair)
        plot_metric_comparison(
            rdf,
            pair,
            fdir / f"pairwise_metric_comparison_{pair_slug}.png",
            title_suffix=pair_title,
        )
        plot_pairwise_relationship_accuracy(
            all_radf,
            pair,
            fdir / f"pairwise_relationship_accuracy_{pair_slug}.png",
        )

    print(f"\n[7] Report...")
    generate_report(rdf, all_gadf, all_radf, all_mcdf, cinfo, ml, outdir / "classifier_report.txt")

    print("\n" + "=" * 70 + "\nDONE\n" + "=" * 70)
    print(f"\nOutput: {outdir}/")
    print("\n--- QUICK SUMMARY (related only group accuracy) ---")
    for metric in METRICS:
        print(f"\n  [{metric_display(metric)}]")
        gc = f'GroupCorrect_{metric}'
        for ms in ml_plot_full:
            df = rdf[rdf['Marker_Set'] == ms].dropna(subset=[f'Pred_{metric}'])
            rel = df[df['Degree'] > 0]
            if len(rel):
                ga = rel[gc].mean() * 100
                dist = rel[rel['Degree'].isin([5, 6])]
                dga = dist[gc].mean() * 100 if len(dist) else 0
                print(f"    {marker_display(ms):<15}: GroupAcc={ga:.1f}%  Distant(5-6th)={dga:.1f}%")

if __name__ == '__main__':
    main()
