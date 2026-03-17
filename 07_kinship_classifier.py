#!/usr/bin/env python3
"""
Kinship Relationship Classifier & Evaluator (Step 7) - v7
==========================================================
Per-metric classifier with ROC-based thresholds (Youden's J):
  - Each metric (IBS, IBD, Kinship) classifies independently
  - Adjacent group pairs: ROC curve -> Youden's J optimal threshold
  - Thresholds chained sequentially for multi-class classification
  - 2nd degree split into Sibling vs GP-GC
  - Grand-Uncle-Nephew = 4th degree

Usage:
  python 07_kinship_classifier.py --results-dir /path/to/06_kinship_analysis
  python 07_kinship_classifier.py --combined-csv all_results_combined.csv
"""

import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({'axes.unicode_minus':False,'figure.dpi':150,
                     'figure.facecolor':'white','font.family':'DejaVu Sans'})

METRICS = ['IBS','IBD','Kinship']
COMPARISON_MARKERS = ['NFS_36K', 'NFS_24K', 'NFS_12K', 'NFS_6K', 'Kintelligence', 'QIAseq']
DISPLAY_METRIC = {'IBS': 'IBS', 'IBD': 'IBD', 'Kinship': 'KCs'}

RELATIONSHIP_TO_DEGREE = {
    'Unrelated':0,'Spouse':0,'Parent-Child':1,
    'Sibling':2,'Grandparent-Grandchild':2,'Half-Sibling':2,
    'Uncle-Nephew':3,'Great-Grandparent':3,
    'Cousin':4,'Grand-Uncle-Nephew':4,
    'Cousin-Once-Removed':5,'Second-Cousin':6,
}
RELATIONSHIP_TO_GROUP = {
    'Unrelated':'G0_Unrelated','Spouse':'G0_Unrelated',
    'Parent-Child':'G1_1st',
    'Sibling':'G2a_Sib','Half-Sibling':'G2a_Sib',
    'Grandparent-Grandchild':'G2b_GPGC',
    'Uncle-Nephew':'G3_3rd','Great-Grandparent':'G3_3rd',
    'Cousin':'G4_4th','Grand-Uncle-Nephew':'G4_4th',
    'Cousin-Once-Removed':'G5_5th','Second-Cousin':'G6_6th',
}
GROUP_ORDER = ['G0_Unrelated','G1_1st','G2a_Sib','G2b_GPGC',
               'G3_3rd','G4_4th','G5_5th','G6_6th']
GROUP_DISPLAY = {
    'G0_Unrelated':'Unrelated','G1_1st':'1st degree',
    'G2a_Sib':'2nd (Sibling)','G2b_GPGC':'2nd (GP-GC)',
    'G3_3rd':'3rd degree','G4_4th':'4th degree',
    'G5_5th':'5th degree','G6_6th':'6th degree',
}
GROUP_DISPLAY_SHORT = {
    'G0_Unrelated':'Unrel','G1_1st':'1st',
    'G2a_Sib':'2nd-Sib','G2b_GPGC':'2nd-GPGC',
    'G3_3rd':'3rd','G4_4th':'4th','G5_5th':'5th','G6_6th':'6th',
}
GROUP_TO_DEGREE = {
    'G0_Unrelated':0,'G1_1st':1,'G2a_Sib':2,'G2b_GPGC':2,
    'G3_3rd':3,'G4_4th':4,'G5_5th':5,'G6_6th':6
}
MARKER_COLORS = {
    'NFS_36K':'#1a5276','NFS_24K':'#2874a6','NFS_20K':'#3498db',
    'NFS_12K':'#e74c3c','NFS_6K':'#9b59b6',
    'Kintelligence':'#27ae60','QIAseq':'#f39c12'
}
GROUP_COLORS = {
    'G0_Unrelated':'#bdc3c7','G1_1st':'#c0392b',
    'G2a_Sib':'#e74c3c','G2b_GPGC':'#f39c12',
    'G3_3rd':'#e67e22','G4_4th':'#f1c40f',
    'G5_5th':'#2ecc71','G6_6th':'#3498db'
}
METRIC_COLORS = {'IBS':'#3498db','IBD':'#e74c3c','Kinship':'#2ecc71'}

def is_nocancer(ms): return 'nocancer' in ms.lower()
def _gd(g): return GROUP_DISPLAY.get(g,g)
def _gs(g): return GROUP_DISPLAY_SHORT.get(g,g)
def _md(metric): return DISPLAY_METRIC.get(metric, metric)

def order_markers(marker_list):
    preferred = ['NFS_36K', 'NFS_24K', 'NFS_12K', 'NFS_6K', 'Kintelligence', 'QIAseq']
    front = [m for m in preferred if m in marker_list]
    tail = sorted([m for m in marker_list if m not in preferred])
    return front + tail

def filter_comparison_markers(marker_list):
    selected = [m for m in marker_list if m in COMPARISON_MARKERS]
    return order_markers(selected) if selected else order_markers(marker_list)


def filter_nfs_markers(marker_list):
    return order_markers([m for m in marker_list if m.startswith('NFS_')])


def normalize_marker_set_names(df):
    """Normalize marker names so comparison plots always include external panels.

    Some combined CSV files use lowercase or variant names (e.g. "qiaseq",
    "kintelligence"). Canonicalizing those names ensures COMPARISON markers are
    detected consistently.
    """
    aliases = {
        'nfs_36k': 'NFS_36K',
        'nfs_24k': 'NFS_24K',
        'nfs_20k': 'NFS_20K',
        'nfs_12k': 'NFS_12K',
        'nfs_6k': 'NFS_6K',
        'kintelligence': 'Kintelligence',
        'qiaseq': 'QIAseq',
    }
    out = df.copy()
    out['Marker_Set'] = out['Marker_Set'].apply(
        lambda x: aliases.get(str(x).strip().lower(), x)
    )
    return out


def normalize_marker_set_names(df):
    """Normalize marker names so comparison plots always include external panels.

    Some combined CSV files use lowercase or variant names (e.g. "qiaseq",
    "kintelligence"). Canonicalizing those names ensures COMPARISON markers are
    detected consistently.
    """
    aliases = {
        'nfs_36k': 'NFS_36K',
        'nfs_24k': 'NFS_24K',
        'nfs_20k': 'NFS_20K',
        'nfs_12k': 'NFS_12K',
        'nfs_6k': 'NFS_6K',
        'kintelligence': 'Kintelligence',
        'qiaseq': 'QIAseq',
    }
    out = df.copy()
    out['Marker_Set'] = out['Marker_Set'].apply(
        lambda x: aliases.get(str(x).strip().lower(), x)
    )
    return out

# ============================================================
# 0. Data Prep
# ============================================================
def fix_degree_labels(df):
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
        for grp in sorted(df['Group'].unique()):
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
        for rel in df['Relationship'].unique():
            s=df[df['Relationship']==rel]; n=len(s)
            pm=s[pc].mode()
            rows.append(dict(Marker_Set=ms,Relationship=rel,
                True_Degree=int(s['Degree'].iloc[0]),True_Group=s['Group'].iloc[0],N=n,
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
            f.write(f"{'='*90}\nCLASSIFICATION THRESHOLDS: {ms}\n{'='*90}\n\n")
            f.write("Method: ROC-based (Youden's J index) per adjacent group pair\n")
            f.write("  threshold = argmax(TPR - FPR) on ROC curve\n")
            f.write("2nd degree split: Sibling vs GP-GC\n\n")
            for metric in METRICS:
                if metric not in minfo: continue
                bd=minfo[metric]['boundaries']; gs=minfo[metric]['stats']
                rd=minfo[metric]['roc_details']
                f.write(f"{'~'*90}\n  METRIC: {metric}\n{'~'*90}\n\n")

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
                csv_rows.append(dict(Metric=metric,Group=grp,GroupLabel=_gs(grp),
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
def plot_accuracy_overall(rdf,marker_list,metric,fdir):
    all_no_nc=order_markers([m for m in marker_list if not is_nocancer(m)])
    nfs_only=order_markers([m for m in all_no_nc if m.startswith('NFS_')])
    pc,gc=f'Pred_{metric}',f'GroupCorrect_{metric}'
    def _summary(mlist):
        rows=[]
        for ms in mlist:
            df=rdf[rdf['Marker_Set']==ms].dropna(subset=[pc])
            n=len(df); nc=int(df[gc].sum())
            rel=df[df['Degree']>0]; nr=len(rel); nrc=int(rel[gc].sum()) if nr else 0
            rows.append(dict(Marker_Set=ms,All=nc/n*100 if n else 0,Related=nrc/nr*100 if nr else 0))
        return pd.DataFrame(rows)
    for ml_sub,suffix,subtitle in [(all_no_nc,'all','All Marker Sets'),(nfs_only,'nfs','NFS Only')]:
        if not ml_sub: continue
        summ=_summary(ml_sub)
        fig,axes=plt.subplots(1,2,figsize=(14,6))
        for ax,col,title in [(axes[0],'All',f'[{_md(metric)}] All Pairs - {subtitle}'),
                             (axes[1],'Related',f'[{_md(metric)}] Related Only - {subtitle}')]:
            order=[m for m in ml_sub if m in summ['Marker_Set'].values]
            vals=[summ[summ['Marker_Set']==m][col].values[0] for m in order]
            colors=[MARKER_COLORS.get(m,'#95a5a6') for m in order]
            bars=ax.bar(range(len(order)),vals,color=colors,edgecolor='white')
            ax.set_xticks(range(len(order))); ax.set_xticklabels(order,rotation=45,ha='right',fontsize=10)
            ax.set_ylabel('Accuracy (%)'); ax.set_title(title,fontweight='bold')
            ax.set_ylim(0,105); ax.grid(axis='y',alpha=.3)
            for b in bars:
                ax.annotate(f'{b.get_height():.1f}%',
                    xy=(b.get_x()+b.get_width()/2,b.get_height()),
                    ha='center',va='bottom',fontsize=9,fontweight='bold')
        plt.tight_layout()
        plt.savefig(fdir/f"accuracy_overall_{metric}_{suffix}.png",dpi=150,bbox_inches='tight',facecolor='white')
        plt.close(); print(f"    Saved: accuracy_overall_{metric}_{suffix}.png")

def plot_accuracy_heatmap_group(gadf,metric,path):
    markers=order_markers(list(gadf['Marker_Set'].unique()))
    groups=[g for g in GROUP_ORDER if g in gadf['Group'].values]
    pivot=gadf.pivot_table(index='Marker_Set',columns='Group',values='GroupAcc',aggfunc='first')
    pivot=pivot.reindex(index=markers,columns=groups)
    fig,ax=plt.subplots(figsize=(max(12,len(groups)*1.8),max(5,len(markers)*0.8)))
    sns.heatmap(pivot,annot=True,fmt='.1f',cmap='RdYlGn',vmin=0,vmax=100,
        linewidths=.5,linecolor='white',ax=ax,cbar_kws={'label':'Accuracy (%)','shrink':.8},
        annot_kws={'fontsize':10,'fontweight':'bold'})
    ax.set_xticklabels([_gd(g) for g in groups],rotation=25,ha='right',fontsize=9)
    ax.set_yticklabels(markers,rotation=0,fontsize=11)
    ax.set_xlabel('Classification Group'); ax.set_ylabel('Marker Set')
    ax.set_title(f'[{_md(metric)}] Group Classification Accuracy',fontsize=13,fontweight='bold')
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print(f"    Saved: {path.name}")

def plot_accuracy_by_relationship(radf,metric,path):
    df=radf[radf['True_Degree']>0].copy()
    if len(df)==0: return
    rel_order=['Parent-Child','Sibling','Grandparent-Grandchild','Uncle-Nephew',
               'Great-Grandparent','Cousin','Grand-Uncle-Nephew','Cousin-Once-Removed','Second-Cousin']
    rels=[r for r in rel_order if r in df['Relationship'].values]
    mlist=order_markers(list(df['Marker_Set'].unique()))
    fig,ax=plt.subplots(figsize=(max(14,len(rels)*2),7))
    nm=len(mlist); bw=0.8/nm; x=np.arange(len(rels))
    for i,ms in enumerate(mlist):
        sub=df[df['Marker_Set']==ms].set_index('Relationship')
        vals=[sub.loc[r,'GroupAcc'] if r in sub.index else 0 for r in rels]
        bars=ax.bar(x+(i-nm/2+.5)*bw,vals,bw,label=ms,
            color=MARKER_COLORS.get(ms,'#95a5a6'),edgecolor='white',linewidth=.3)
        for b in bars:
            h=b.get_height()
            if h>0:
                ax.annotate(f'{h:.0f}',xy=(b.get_x()+b.get_width()/2,h),
                    ha='center',va='bottom',fontsize=6,fontweight='bold',rotation=90)
    ax.set_xticks(x)
    grpmap={r:df[df['Relationship']==r]['True_Group'].iloc[0] for r in rels}
    ax.set_xticklabels([f'{r}\n({_gs(grpmap.get(r,""))})' for r in rels],rotation=0,ha='center',fontsize=8)
    ax.set_ylabel('Group Classification Accuracy (%)'); ax.set_ylim(0,120)
    ax.set_title(f'[{_md(metric)}] Classification Accuracy by Relationship',fontsize=13,fontweight='bold')
    ax.legend(fontsize=8,ncol=2,loc='upper right'); ax.grid(axis='y',alpha=.3)
    ax.axhline(100,color='gray',ls='--',alpha=.3)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print(f"    Saved: {path.name}")

def plot_confusion_matrices(rdf,mlist,metric,outdir):
    outdir=Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    pc=f'Pred_{metric}'
    for ms in mlist:
        df=rdf[rdf['Marker_Set']==ms].dropna(subset=[pc])
        if len(df)==0: continue
        yt=df['Group'].astype(str); yp=df[pc].astype(str)
        labels=[g for g in GROUP_ORDER if g in set(yt)|set(yp)]
        cm=confusion_matrix(yt,yp,labels=labels)
        cmn=cm.astype(float)/np.maximum(cm.sum(axis=1,keepdims=True),1)*100
        tl=[_gd(l) for l in labels]
        fig,axes=plt.subplots(1,2,figsize=(18,7))
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=axes[0],
            xticklabels=tl,yticklabels=tl,linewidths=.5,linecolor='white')
        axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('True'); axes[0].set_title('Counts',fontweight='bold')
        plt.setp(axes[0].get_xticklabels(),rotation=30,ha='right',fontsize=8)
        plt.setp(axes[0].get_yticklabels(),rotation=0,fontsize=8)
        sns.heatmap(cmn,annot=True,fmt='.1f',cmap='RdYlGn',vmin=0,vmax=100,ax=axes[1],
            xticklabels=tl,yticklabels=tl,linewidths=.5,linecolor='white')
        axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('True'); axes[1].set_title('Row-normalized (%)',fontweight='bold')
        plt.setp(axes[1].get_xticklabels(),rotation=30,ha='right',fontsize=8)
        plt.setp(axes[1].get_yticklabels(),rotation=0,fontsize=8)
        plt.suptitle(f'[{_md(metric)}] Confusion Matrix - {ms}',fontsize=14,fontweight='bold',y=1.02)
        plt.tight_layout()
        plt.savefig(outdir/f"confusion_{metric}_{ms}.png",dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
        print(f"    Saved: confusion_{metric}_{ms}.png")

def plot_misclassification_summary(mcdf,metric,path,include_markers=None):
    if len(mcdf)==0: print("    No misclassifications."); return
    mcdf_f=mcdf[~mcdf['Marker_Set'].apply(is_nocancer)]
    if include_markers is not None:
        mcdf_f=mcdf_f[mcdf_f['Marker_Set'].isin(include_markers)]
    if len(mcdf_f)==0: print("    No misclassifications."); return
    pivot=mcdf_f.groupby(['Marker_Set','True_Group']).size().reset_index(name='Errors')
    mlist=order_markers(list(pivot['Marker_Set'].unique()))
    groups=[g for g in GROUP_ORDER if g in pivot['True_Group'].values]
    fig,ax=plt.subplots(figsize=(max(12,len(groups)*2),6))
    nm=len(mlist); bw=0.8/nm; x=np.arange(len(groups))
    for i,ms in enumerate(mlist):
        sub=pivot[pivot['Marker_Set']==ms].set_index('True_Group')
        vals=[sub.loc[g,'Errors'] if g in sub.index else 0 for g in groups]
        bars=ax.bar(x+(i-nm/2+.5)*bw,vals,bw,label=ms,
            color=MARKER_COLORS.get(ms,'#95a5a6'),edgecolor='white',linewidth=.3)
        for b in bars:
            h=b.get_height()
            if h>0: ax.annotate(f'{int(h)}',xy=(b.get_x()+b.get_width()/2,h),
                ha='center',va='bottom',fontsize=7,fontweight='bold')
    ax.set_xticks(x); ax.set_xticklabels([_gd(g) for g in groups],fontsize=9,rotation=15,ha='right')
    ax.set_ylabel('# Misclassified Pairs')
    ax.set_title(f'[{_md(metric)}] Misclassification by True Group',fontsize=12,fontweight='bold')
    ax.legend(fontsize=9,ncol=2); ax.grid(axis='y',alpha=.3)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print(f"    Saved: {path.name}")

def plot_forensic_scenarios(rdf,mlist,metric,path):
    mlist=[m for m in mlist if not is_nocancer(m)]
    gc,w1,pc = f'GroupCorrect_{metric}',f'DegWithin1_{metric}',f'Pred_{metric}'
    fig,axes=plt.subplots(2,1,figsize=(16,12))
    ax=axes[0]
    all_groups=[g for g in GROUP_ORDER if g!='G0_Unrelated']
    for ms in mlist:
        df=rdf[(rdf['Marker_Set']==ms)&(rdf['Degree']>0)].dropna(subset=[pc])
        accs=[df[df['Group']==g][gc].mean()*100 if len(df[df['Group']==g]) else 0 for g in all_groups]
        ax.plot(range(len(all_groups)),accs,'o-',label=ms,color=MARKER_COLORS.get(ms,'gray'),lw=2,ms=8)
    ax.set_xticks(range(len(all_groups)))
    ax.set_xticklabels([_gd(g) for g in all_groups],rotation=15,ha='right',fontsize=9)
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0,105); ax.grid(alpha=.3)
    ax.set_title(f'[{_md(metric)}] Group Accuracy by KC Distance',fontsize=14,fontweight='bold')
    ax.legend(fontsize=9,ncol=2)
    ax=axes[1]; da=[]
    for ms in mlist:
        df=rdf[(rdf['Marker_Set']==ms)&(rdf['Degree'].isin([5,6]))].dropna(subset=[pc])
        da.append(dict(Marker_Set=ms,Exact=df[gc].mean()*100 if len(df) else 0,
                       Within1=df[w1].mean()*100 if len(df) else 0))
    da=pd.DataFrame(da); x=np.arange(len(da)); w=0.35
    b1=ax.bar(x-w/2,da['Exact'],w,label='Exact',color='#3498db',edgecolor='white')
    b2=ax.bar(x+w/2,da['Within1'],w,label='Within +/-1',color='#2ecc71',edgecolor='white')
    ax.set_xticks(x); ax.set_xticklabels(da['Marker_Set'],rotation=45,ha='right')
    ax.set_ylabel('Accuracy (%)'); ax.set_ylim(0,110); ax.grid(axis='y',alpha=.3)
    ax.set_title(f'[{_md(metric)}] Extended KC (5-6th)',fontsize=14,fontweight='bold'); ax.legend(fontsize=11)
    for b in list(b1)+list(b2):
        ax.annotate(f'{b.get_height():.0f}%',xy=(b.get_x()+b.get_width()/2,b.get_height()),
            ha='center',va='bottom',fontsize=9,fontweight='bold')
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print(f"    Saved: {path.name}")

def plot_thresholds(cinfo,ms,metric,path):
    """Threshold plot: tilted red values on x-axis."""
    if metric not in cinfo[ms]: return
    bd=cinfo[ms][metric]['boundaries']; gs=cinfo[ms][metric]['stats']
    grps=[g for g in GROUP_ORDER if g in gs]
    fig,ax=plt.subplots(figsize=(10,max(5,len(grps)*0.8)))
    for i,grp in enumerate(grps):
        s=gs[grp]; c=GROUP_COLORS.get(grp,'#95a5a6')
        ax.barh(i,s['q75']-s['q25'],left=s['q25'],height=.6,color=c,alpha=.6,edgecolor='black',lw=.5)
        ax.plot(s['median'],i,'k|',ms=15,mew=2)
        ax.plot([s['min'],s['max']],[i,i],'k-',lw=.5,alpha=.5)
        ax.annotate(f'{s["median"]:.4f}',xy=(s['median'],i+0.35),
            ha='center',va='bottom',fontsize=7,color='black',fontweight='bold')
    drawn=set()
    for grp,(lo,hi) in bd.items():
        for th in [lo,hi]:
            if th not in (float('inf'),float('-inf')):
                th_r=round(th,8)
                if th_r not in drawn:
                    ax.axvline(th,color='red',ls='--',alpha=.6,lw=1.2)
                    ax.annotate(f'{th:.4f}',xy=(th,-0.7),ha='center',va='top',
                        fontsize=7,color='red',fontweight='bold',rotation=45)
                    drawn.add(th_r)
    ax.set_yticks(range(len(grps)))
    ax.set_yticklabels([_gd(g) for g in grps],fontsize=9)
    ax.set_xlabel(_md(metric),fontsize=12)
    ax.set_title(f'Classification Threshold of {ms} with {_md(metric)}',fontsize=12,fontweight='bold')
    ax.grid(axis='x',alpha=.3); ax.set_ylim(-1.5,len(grps)-0.3)
    plt.tight_layout(); plt.savefig(path,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
    print(f"    Saved: {path.name}")

def plot_metric_comparison(rdf,marker_list,path,title_suffix=''):
    mlist=order_markers([m for m in marker_list if not is_nocancer(m) and m!='NFS_20K'])
    rows=[]
    for ms in mlist:
        df=rdf[rdf['Marker_Set']==ms]
        for metric in METRICS:
            gc=f'GroupCorrect_{metric}'
            if gc not in df.columns: continue
            valid=df.dropna(subset=[f'Pred_{metric}']); rel=valid[valid['Degree']>0]
            rows.append(dict(Marker_Set=ms,Metric=metric,
                All=valid[gc].mean()*100 if len(valid) else 0,
                Related=rel[gc].mean()*100 if len(rel) else 0))
    mdf=pd.DataFrame(rows)
    if len(mdf)==0: return
    fig,axes=plt.subplots(1,2,figsize=(16,6.5))
    for ax,col,title in zip(axes,['All','Related'],['All Pairs','Related Only']):
        pv=mdf.pivot_table(index='Marker_Set',columns='Metric',values=col)
        pv=pv.reindex(mlist)
        x=np.arange(len(pv)); w=0.25
        for i,m in enumerate(METRICS):
            if m in pv.columns:
                bars=ax.bar(x+(i-1)*w,pv[m],w,label=_md(m),color=METRIC_COLORS[m],edgecolor='white')
                for b in bars:
                    ax.annotate(f'{b.get_height():.1f}',
                        xy=(b.get_x()+b.get_width()/2,b.get_height()+0.5),
                        ha='center',va='bottom',fontsize=7,fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(pv.index,rotation=45,ha='right')
        ax.set_ylabel('Group Accuracy (%)')
        suffix=f' - {title_suffix}' if title_suffix else ''
        ax.set_title(f'Metric Comparison ({title}){suffix}',fontweight='bold')
        ax.set_ylim(0,110)
        ax.grid(axis='y',alpha=.3)
        ax.legend(loc='upper center',bbox_to_anchor=(0.5,1.0),ncol=3,frameon=True,borderaxespad=0.3)
    plt.tight_layout(rect=[0,0,1,0.98]); plt.savefig(path,dpi=150,bbox_inches='tight',facecolor='white'); plt.close()
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

def generate_report(rdf,all_gadf,all_radf,all_mcdf,cinfo,mlist,rpath):
    with open(rpath,'w',encoding='utf-8') as f:
        f.write("="*100+"\nKINSHIP CLASSIFIER - EVALUATION REPORT (v7)\n"+"="*100+"\n\n")
        f.write("METHOD\n"+"-"*80+"\n")
        f.write("  Per-metric classification (IBS, IBD, KCs independently)\n")
        f.write("  ROC-based thresholds: Youden's J (argmax TPR-FPR) per adjacent group pair\n")
        f.write("  2nd degree split: Sibling vs GP-GC\n")
        f.write("  Grand-Uncle-Nephew = 4th degree\n\n")
        for metric in METRICS:
            gadf=all_gadf[metric]; radf=all_radf[metric]; mcdf=all_mcdf[metric]
            f.write(f"\n{'#'*100}\n  METRIC: {_md(metric)}\n{'#'*100}\n\n")
            f.write(f"  1. SUMMARY\n  "+"-"*70+"\n\n")
            for ms in mlist:
                gc,dc=f'GroupCorrect_{metric}',f'DegCorrect_{metric}'
                df=rdf[rdf['Marker_Set']==ms].dropna(subset=[f'Pred_{metric}'])
                n=len(df); nc=int(df[gc].sum()); ndc=int(df[dc].sum())
                rel=df[df['Degree']>0]; nr=len(rel)
                nrc=int(rel[gc].sum()) if nr else 0; nrdc=int(rel[dc].sum()) if nr else 0
                f.write(f"    [{ms}]\n")
                f.write(f"      All:     {n:>6} | GroupAcc: {nc/n*100:>5.1f}% | DegreeAcc: {ndc/n*100:>5.1f}%\n")
                if nr: f.write(f"      Related: {nr:>6} | GroupAcc: {nrc/nr*100:>5.1f}% | DegreeAcc: {nrdc/nr*100:>5.1f}%\n")
                f.write("\n")

            # ROC details
            f.write(f"\n  2. ROC THRESHOLDS\n  "+"-"*70+"\n")
            for ms in mlist:
                if metric not in cinfo.get(ms,{}): continue
                rd=cinfo[ms][metric]['roc_details']
                f.write(f"\n    [{ms}]\n")
                f.write(f"    {'Pair':<30} {'ROC_Th':>10} {'AUC':>8} {'Midpoint':>10}\n")
                f.write(f"    "+"-"*60+"\n")
                for (gHi,gLo),det in rd.items():
                    pair_name=f"{_gs(gHi)} vs {_gs(gLo)}"
                    auc_s=f"{det['auc']:.3f}" if det['auc'] is not None else "N/A"
                    f.write(f"    {pair_name:<30} {det['threshold']:>10.6f} {auc_s:>8} {det['midpoint']:>10.6f}\n")

            f.write(f"\n\n  3. PER-GROUP ACCURACY\n  "+"-"*70+"\n")
            for ms in mlist:
                f.write(f"\n    [{ms}]\n    {'Group':<22} {'N':>5} {'GrpAcc':>7} {'DegAcc':>7} {'+/-1':>6}\n    "+"-"*50+"\n")
                for _,r in gadf[gadf['Marker_Set']==ms].sort_values('Group').iterrows():
                    f.write(f"    {r['Label']:<22} {int(r['N']):>5} {r['GroupAcc']:>6.1f}% {r['DegreeAcc']:>6.1f}% {r['Within1']:>5.1f}%\n")
            f.write(f"\n\n  4. PER-RELATIONSHIP ACCURACY\n  "+"-"*70+"\n")
            for ms in mlist:
                f.write(f"\n    [{ms}]\n    {'Relationship':<26} {'Group':>10} {'N':>5} {'GrpAcc':>7} {'DegAcc':>7} {'->':>10}\n    "+"-"*70+"\n")
                sub=radf[(radf['Marker_Set']==ms)&(radf['True_Degree']>0)].sort_values('True_Group')
                for _,r in sub.iterrows():
                    f.write(f"    {r['Relationship']:<26} {_gs(r['True_Group']):>10} {int(r['N']):>5} "
                            f"{r['GroupAcc']:>6.1f}% {r['DegreeAcc']:>6.1f}% ->{_gs(r['MostPredGroup']):>9}\n")
            f.write(f"\n\n  5. MISCLASSIFIED (related)\n  "+"-"*70+"\n")
            if len(mcdf)==0: f.write("    None!\n")
            else:
                for ms in mlist:
                    sub=mcdf[mcdf['Marker_Set']==ms]
                    if len(sub)==0: f.write(f"\n    [{ms}] None\n"); continue
                    f.write(f"\n    [{ms}] {len(sub)} misclassified\n")
                    f.write(f"    {'Fam':>4} {'S1':<16} {'S2':<16} {'Relation':<22} {'True':>10} {'Pred':>10} {metric:>10}\n    "+"-"*95+"\n")
                    for _,r in sub.sort_values(['True_Group','Family']).iterrows():
                        mv=f"{r['Metric_Value']:.5f}" if pd.notna(r.get('Metric_Value')) else 'N/A'
                        f.write(f"    {r['Family']:>4} {str(r['Sample1']):<16} {str(r['Sample2']):<16} "
                                f"{r['True_Relationship']:<22} {_gs(r['True_Group']):>10} -> {_gs(r['Pred_Group']):>8} {mv:>10}\n")
        f.write("\n\n"+"="*100+"\nEND OF REPORT\n"+"="*100+"\n")
    print(f"    Saved: {rpath.name}")

# ============================================================
# Main
# ============================================================
def main():
    parser=argparse.ArgumentParser(description='Kinship Classifier v7 (per-metric, ROC Youden)')
    parser.add_argument('--results-dir',type=str)
    parser.add_argument('--combined-csv',type=str)
    parser.add_argument('--output-dir',type=str,default=None)
    args=parser.parse_args()
    if args.combined_csv: cpath=Path(args.combined_csv)
    elif args.results_dir: cpath=Path(args.results_dir)/"all_results_combined.csv"
    else: cpath=Path.home()/"kinship/Analysis/20251031_wgrs/06_kinship_analysis/all_results_combined.csv"
    if not cpath.exists(): print(f"ERROR: {cpath} not found"); sys.exit(1)

    print("="*70+"\nKINSHIP CLASSIFIER v7 (per-metric, ROC-based thresholds)\n"+"="*70)
    print(f"\n[1] Loading: {cpath}")
    all_df=pd.read_csv(cpath)
    all_df=normalize_marker_set_names(all_df)
    marker_list=order_markers(sorted(all_df['Marker_Set'].unique()))
    ml=order_markers([m for m in marker_list if not is_nocancer(m)])
    ml_comp=filter_comparison_markers(ml)
    ml_nfs=filter_nfs_markers(ml_comp)
    print(f"    {len(all_df):,} rows, {len(marker_list)} markers ({len(ml)} non-nocancer)")
    print(f"    Comparison markers: {', '.join(ml_comp)}")
    print(f"    NFS-only markers: {', '.join(ml_nfs)}")
    print(f"\n[1b] Fixing degrees & adding groups...")
    all_df=fix_degree_labels(all_df)

    outdir=(Path(args.output_dir) if args.output_dir else
            Path(args.results_dir)/"classifier" if args.results_dir else cpath.parent/"classifier")
    fdir,tdir=outdir/"figures",outdir/"tables"
    for d in [outdir,fdir,tdir]: d.mkdir(parents=True,exist_ok=True)

    print(f"\n[2] Classifying (per-metric, ROC Youden's J)...")
    rdf,cinfo=run_classifier(all_df,ml)

    print(f"\n[3] Evaluating...")
    all_gadf,all_radf,all_mcdf={},{},{}
    for metric in METRICS:
        print(f"\n  --- {metric} ---")
        gadf=compute_group_accuracy(rdf,ml,metric); gadf.to_csv(tdir/f"accuracy_group_{metric}.csv",index=False)
        radf=compute_rel_accuracy(rdf,ml,metric); radf.to_csv(tdir/f"accuracy_rel_{metric}.csv",index=False)
        mcdf=find_misclassified(rdf,ml,metric); mcdf.to_csv(tdir/f"misclassified_{metric}.csv",index=False)
        print(f"    Misclassified (related): {len(mcdf)}")
        all_gadf[metric]=gadf; all_radf[metric]=radf; all_mcdf[metric]=mcdf

    print(f"\n[4] Thresholds..."); export_thresholds(cinfo,outdir)
    print(f"\n[5] Master tables..."); generate_master_tables(rdf,ml,tdir)

    print(f"\n[6] Figures...")
    marker_variants=[('comparison', ml_comp, '', ''), ('nfs_only', ml_nfs, '_nfs_only', 'NFS only')]
    for metric in METRICS:
        print(f"\n  --- {metric} ---")
        plot_accuracy_overall(rdf,ml_comp,metric,fdir)
        for variant_name, mlist, file_suffix, title_suffix in marker_variants:
            if len(mlist) <= 1:
                continue
            gadf_sub=all_gadf[metric][all_gadf[metric]['Marker_Set'].isin(mlist)]
            radf_sub=all_radf[metric][all_radf[metric]['Marker_Set'].isin(mlist)]
            plot_accuracy_heatmap_group(gadf_sub,metric,fdir/f"heatmap_group_{metric}{file_suffix}.png")
            plot_accuracy_by_relationship(radf_sub,metric,fdir/f"accuracy_rel_{metric}{file_suffix}.png")
            conf_dir = fdir if variant_name=='comparison' else fdir/"nfs_only"
            plot_confusion_matrices(rdf,mlist,metric,conf_dir)
            plot_misclassification_summary(all_mcdf[metric],metric,fdir/f"misclass_{metric}{file_suffix}.png",include_markers=mlist)
            plot_forensic_scenarios(rdf,mlist,metric,fdir/f"forensic_{metric}{file_suffix}.png")
        for ms in ml:
            plot_thresholds(cinfo,ms,metric,fdir/f"thresholds_{metric}_{ms}.png")

    print(f"\n  --- Metric comparison ---")
    for _, mlist, file_suffix, title_suffix in marker_variants:
        if len(mlist) <= 1:
            continue
        plot_metric_comparison(rdf,mlist,fdir/f"metric_comparison{file_suffix}.png",title_suffix=title_suffix)

    pair_plots=[
        (['NFS_12K','Kintelligence'],'12K vs Kintelligence','metric_comparison_12k_vs_kintelligence'),
        (['NFS_6K','QIAseq'],'6K vs QIAseq','metric_comparison_6k_vs_qiaseq')
    ]
    for pair,title,base_fn in pair_plots:
        available=[m for m in pair if m in ml_comp]
        if len(available)==2:
            plot_metric_comparison(rdf,available,fdir/f"{base_fn}.png",title_suffix=title)
            available_nfs=[m for m in available if m in ml_nfs]
            if len(available_nfs)>=1:
                plot_metric_comparison(rdf,available_nfs,fdir/f"{base_fn}_nfs_only.png",title_suffix=f"{title} (NFS only)")

    print(f"\n[7] Report...")
    generate_report(rdf,all_gadf,all_radf,all_mcdf,cinfo,ml,outdir/"classifier_report.txt")

    print("\n"+"="*70+"\nDONE\n"+"="*70)
    print(f"\nOutput: {outdir}/")
    print("\n--- QUICK SUMMARY ---")
    for metric in METRICS:
        print(f"\n  [{metric}]")
        for ms in ml:
            gc=f'GroupCorrect_{metric}'
            df=rdf[rdf['Marker_Set']==ms].dropna(subset=[f'Pred_{metric}'])
            rel=df[df['Degree']>0]
            if len(rel):
                ga=rel[gc].mean()*100
                dist=rel[rel['Degree'].isin([5,6])]
                dga=dist[gc].mean()*100 if len(dist) else 0
                print(f"    {ms:<15}: GroupAcc={ga:.1f}%  Distant(5-6th)={dga:.1f}%")

if __name__=='__main__':
    main()
