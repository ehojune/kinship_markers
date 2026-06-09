#!/usr/bin/env python3
"""
Step 5: Evaluate kinship results
================================
Load existing Step 3 PLINK/KING outputs plus Step 4 ground truth, then generate
combined result tables, ROC metrics, figures, and reports. Outputs are written
under 07_evalutate_kinship by default. This step does not rerun PLINK or KING.
"""

import argparse
from pathlib import Path

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

HOME = Path.home()
DEFAULT_WORK_DIR  = "/mnt/d/Research/20251031_wgrs"
DEFAULT_JOINT_VCF = "/mnt/d/Research/20251031_wgrs/05_jointcall/joint_called.allsites.vcf.gz"
DEFAULT_ANALYSIS_DIR = "/mnt/d/Research/20251031_wgrs/06_kinship_analysis"

# ============================================================
# Plotting Constants
# ============================================================
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.facecolor'] = 'white'

MARKER_COLORS = {
    'NFS_36K': '#1a5276', 'NFS_24K': '#2874a6',
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
    'Parent-Child': 'Parent-Child\n(1)',
    'Sibling': 'Sibling\n(2)',
    'Grandparent-Grandchild': 'Grandparent\n(2)',
    'Uncle-Nephew': 'Uncle-Nephew\n(3)',
    'Cousin': 'Cousin\n(4)',
    'Grand-Uncle-Nephew': 'Grand-Uncle\n(4)',
    'Cousin-Once-Removed': 'Cousin-1R\n(5)',
    'Second-Cousin': '2nd-Cousin\n(6)',
    'Spouse': 'Spouse\n(0)',
    'Unrelated': 'Unrelated\n(0)'
}

COMPARISON_MARKERS = ['NFS_36K', 'NFS_24K', 'NFS_12K', 'NFS_6K', 'Kintelligence', 'QIAseq']
DISPLAY_METRIC = {'IBS': 'IBS', 'IBD': 'IBD', 'Kinship': 'KCs'}


def collapse_spouse_to_others(df):
    df = df.copy()
    df['Relationship'] = df['Relationship'].replace({'Spouse': 'Unrelated'})
    return df


def filter_comparison_markers(marker_list):
    selected = [m for m in marker_list if m in COMPARISON_MARKERS]
    return selected if selected else marker_list


def filter_nfs_markers(marker_list):
    return [m for m in marker_list if m.startswith('NFS_')]


def _md(metric):
    return DISPLAY_METRIC.get(metric, metric)
def load_plink_genome(filepath):
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath, sep=r'\s+')
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
        return 'Unrelated\n(0)' if row['Degree'] == 0 else f"{row['Degree']}"
    df['DL'] = df.apply(get_label, axis=1)
    order = ['1','2','3','4','5','6','Unrelated\n(0)']
    avail = [o for o in order if o in df['DL'].values]
    pal = []
    for o in avail:
        for d in range(1, 7):
            if f'{d}' in o:
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
        ax.set_xlabel('Degree', fontsize=12); ax.set_ylabel(_md(m), fontsize=12)
        ax.set_title(f'{_md(m)}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45); ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f'{marker_set} - Distribution by Degree', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"    Saved: {output_path.name}")

def plot_boxplot_by_relationship(all_df, marker_set, output_path):
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    df = collapse_spouse_to_others(df)
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
        ax.set_xlabel('Relationship', fontsize=12); ax.set_ylabel(_md(m), fontsize=12)
        ax.set_title(f'{_md(m)}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=9); ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f'{marker_set} - Distribution by Relationship', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()
    print(f"    Saved: {output_path.name}")

def plot_violin_by_relationship(all_df, marker_set, output_path):
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    df = collapse_spouse_to_others(df)
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
        ax.set_xlabel('Relationship', fontsize=12); ax.set_ylabel(_md(m), fontsize=12)
        ax.set_title(f'{_md(m)}', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=45, labelsize=9); ax.grid(axis='y', alpha=0.3)
    plt.suptitle(f'{marker_set} - Violin by Relationship', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()


def plot_relationship_distribution_single(all_df, marker_set, metric, output_path, kind='violin'):
    """Single-metric relationship distribution plot for readability with long labels."""
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    df = collapse_spouse_to_others(df)
    if len(df) == 0:
        return
    rel_order = [r for r in RELATIONSHIP_ORDER if r in df['Relationship'].values]
    if not rel_order:
        return
    df['RL'] = df['Relationship'].map(lambda x: RELATIONSHIP_LABELS.get(x, x))
    lo = [RELATIONSHIP_LABELS.get(r, r) for r in rel_order]
    pal = [RELATIONSHIP_COLORS.get(r, '#95a5a6') for r in rel_order]
    data = df.dropna(subset=[metric])
    if len(data) == 0:
        return

    fig, ax = plt.subplots(figsize=(max(10, len(lo) * 1.15), 6))
    if kind == 'box':
        sns.boxplot(data=data, x='RL', y=metric, order=lo, palette=pal, ax=ax, width=0.6, linewidth=1.5)
        sns.stripplot(data=data, x='RL', y=metric, order=lo, color='black', size=2, alpha=0.3, ax=ax, jitter=True)
    else:
        sns.violinplot(data=data, x='RL', y=metric, order=lo, palette=pal, ax=ax, inner='box', linewidth=1)
    ax.set_xlabel('Relationship', fontsize=12)
    ax.set_ylabel(_md(metric), fontsize=12)
    ax.set_title(f'{marker_set} - {_md(metric)} ({kind.capitalize()} by Relationship)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=30, labelsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    Saved: {output_path.name}")

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
                cbar_kws={'shrink': 0.6, 'label': _md(metric)}, ax=ax)
    labels = [f"{s.split('-')[1]}-{s.split('-')[2]}" if len(s.split('-')) >= 3 else s for s in samples]
    fs = max(4, min(8, 120 // n))
    ax.set_xticks(np.arange(n) + 0.5); ax.set_yticks(np.arange(n) + 0.5)
    ax.set_xticklabels(labels, rotation=90, ha='center', fontsize=fs)
    ax.set_yticklabels(labels, rotation=0, ha='right', fontsize=fs)
    ax.set_title(f'{marker_set} - {_md(metric)}', fontsize=14, fontweight='bold')
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
                cbar_kws={'shrink': 0.7, 'label': _md(metric)}, ax=ax)
    labels = [s.split('-')[-1] for s in samples]
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(labels, rotation=0, fontsize=11)
    ax.set_title(f'Family {family} - {marker_set} - {_md(metric)}', fontsize=13, fontweight='bold')
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()


# ============================================================
# ROC calculation (all 13 scenarios)
# ============================================================
def calculate_roc_metrics(y_true, y_score):
    y_score = pd.to_numeric(y_score, errors='coerce')
    y_score = np.asarray(y_score, dtype=float)
    valid = ~np.isnan(y_score)
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
        'close_vs_unrelated':         {'pos': lambda d: d['Degree'].isin([1,2,3,4]), 'neg': lambda d: d['Is_Related']==False, 'desc': '1-4 vs Unrelated'},
        'distant_vs_unrelated':       {'pos': lambda d: d['Degree'].isin([5,6]), 'neg': lambda d: d['Is_Related']==False, 'desc': '5-6 vs Unrelated'},
        '1st_vs_2nd':  {'pos': lambda d: d['Degree']==1, 'neg': lambda d: d['Degree']==2, 'desc': '1 vs 2'},
        '2nd_vs_3rd':  {'pos': lambda d: d['Degree']==2, 'neg': lambda d: d['Degree']==3, 'desc': '2 vs 3'},
        '3rd_vs_4th':  {'pos': lambda d: d['Degree']==3, 'neg': lambda d: d['Degree']==4, 'desc': '3 vs 4'},
        '4th_vs_5th':  {'pos': lambda d: d['Degree']==4, 'neg': lambda d: d['Degree']==5, 'desc': '4 vs 5'},
        '5th_vs_6th':  {'pos': lambda d: d['Degree']==5, 'neg': lambda d: d['Degree']==6, 'desc': '5 vs 6'},
        '4th_vs_unrelated':  {'pos': lambda d: d['Degree']==4, 'neg': lambda d: d['Is_Related']==False, 'desc': '4 vs Unrelated'},
        '5th_vs_unrelated':  {'pos': lambda d: d['Degree']==5, 'neg': lambda d: d['Is_Related']==False, 'desc': '5 vs Unrelated'},
        '6th_vs_unrelated':  {'pos': lambda d: d['Degree']==6, 'neg': lambda d: d['Is_Related']==False, 'desc': '6 vs Unrelated'},
        '12345_vs_6':  {'pos': lambda d: d['Degree'].isin([1,2,3,4,5]), 'neg': lambda d: d['Degree']==6, 'desc': '1-5 vs 6'},
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
        ax.set_title(f'{_md(metric)}', fontsize=14, fontweight='bold')
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
    fig, ax = plt.subplots(figsize=(18, max(5, len(mo) * 1.15 + 2)))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', vmin=0.5, vmax=1.0,
                ax=ax, linewidths=0.5, cbar_kws={'label':'AUC','shrink':0.8}, annot_kws={'size':9})
    ax.set_title(f'{_md(metric)} - AUC by Scenario', fontsize=14, fontweight='bold')
    ax.set_xlabel('Scenario'); ax.set_ylabel('Marker Set')
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=20, ha='right', fontsize=10)
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
        ax.set_xlabel('Adjacent ()', fontsize=12); ax.set_ylabel('AUC', fontsize=12)
        ax.set_title(f'{_md(metric)}', fontsize=14, fontweight='bold')
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
                       label=f"{deg}" if deg>0 else "Unrel", alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
        valid = data[['Expected_Kinship', m]].dropna()
        if len(valid) > 2:
            corr = valid['Expected_Kinship'].corr(valid[m])
            ax.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, fontweight='bold')
        if m == 'Kinship':
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
            ax.plot(lims, lims, 'k--', alpha=0.5)
        ax.set_xlabel('Expected KCs'); ax.set_ylabel(f'Observed {_md(m)}')
        ax.set_title(f'{_md(m)}', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=8, ncol=2); ax.grid(alpha=0.3)
    plt.suptitle(f'{marker_set} - Expected vs Observed', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(); plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white'); plt.close()


# ============================================================
# NEW: Marker Comparison Overlay
# ============================================================
def plot_marker_comparison_overlay(all_df, marker_list, metric, output_path, title_suffix=''):
    """All markers side-by-side per degree for a given metric."""
    related = all_df[all_df['Degree'] > 0].copy()
    if len(related) == 0: return
    related['DL'] = related['Degree'].apply(lambda d: f"{d}")
    degrees = sorted(related['Degree'].unique())
    fig, ax = plt.subplots(figsize=(max(14, len(degrees)*2.5), 7))
    data = related.dropna(subset=[metric])
    if len(data) == 0: plt.close(); return
    sns.boxplot(data=data, x='DL', y=metric, hue='Marker_Set',
                order=[f"{d}" for d in degrees], hue_order=marker_list, ax=ax,
                palette={m: MARKER_COLORS.get(m,'gray') for m in marker_list}, width=0.8, linewidth=1)
    ax.set_xlabel('Degree ()', fontsize=13); ax.set_ylabel(_md(metric), fontsize=13)
    suffix = f' ({title_suffix})' if title_suffix else ''
    ax.set_title(f'Marker Comparison - {_md(metric)}{suffix}', fontsize=15, fontweight='bold')
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
    axes[0].set_xlabel('Degree ()'); axes[0].set_ylabel('KCs (Mean±Std)')
    axes[0].set_title('KCs by Degree', fontweight='bold'); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('Degree ()'); axes[1].set_ylabel('CV')
    axes[1].set_title('KCs Variability', fontweight='bold'); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
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
        ax.set_xlabel('Adjacent Degrees ()'); ax.set_ylabel("Cohen's d")
        ax.set_title(f'{_md(metric)}', fontsize=14, fontweight='bold')
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
    plt.suptitle(f'Confusion Matrix: Related vs Unrelated ({_md(metric)})', fontsize=14, fontweight='bold', y=1.05)
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
            label = f"{deg}" if deg > 0 else "Unrelated"
            f.write(f"  {label:<15}: {int(row['Sample1']):>6} pairs  (phi={row['Expected_Kinship']:.4f})\n")
        f.write("\n4. CLASSIFICATION PERFORMANCE (AUC)\n" + "-"*80 + "\n")
        for sn in ['related_vs_unrelated','close_vs_unrelated','distant_vs_unrelated',
                    '4th_vs_unrelated','5th_vs_unrelated','6th_vs_unrelated','12345_vs_6']:
            sdata = roc_results[roc_results['Scenario']==sn]
            if len(sdata) == 0: continue
            desc = sdata['Description'].iloc[0]
            f.write(f"\n  [{desc}]\n  {'Marker':<15} {'IBS':>10} {'IBD':>10} {'KCs':>10}\n  " + "-"*50 + "\n")
            for mk in marker_list:
                vals = {}
                for m in ['IBS','IBD','Kinship']:
                    r = sdata[(sdata['Marker_Set']==mk) & (sdata['Metric']==m)]
                    vals[m] = f"{r['AUC'].values[0]:.4f}" if len(r)>0 and pd.notna(r['AUC'].values[0]) else "N/A"
                f.write(f"  {mk:<15} {vals['IBS']:>10} {vals['IBD']:>10} {vals['Kinship']:>10}\n")
        f.write("\n\n5. OPTIMAL THRESHOLDS (Youden's J)\n" + "-"*80 + "\n")
        td = roc_results[roc_results['Scenario']=='related_vs_unrelated']
        f.write(f"\n  [Related vs Unrelated]\n  {'Marker':<15} {'IBS':>12} {'IBD':>12} {'KCs':>12}\n  " + "-"*55 + "\n")
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
    analysis_dir = Path(args.analysis_dir)
    eval_dir = Path(args.eval_dir)
    results_dir = analysis_dir / "results"
    fig_dir = eval_dir / "figures"
    reports_dir = eval_dir / "reports"
    DD = fig_dir/"distributions"; DH = fig_dir/"heatmaps"; DR = fig_dir/"roc_curves"
    DS = fig_dir/"scatter"; DC = fig_dir/"comparison"; DM = fig_dir/"summary"
    for d in [DD, DH, DR, DS, DC, DM, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
    if gt_df is None:
        gt_path = analysis_dir / "family_relationships.csv"
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
    comparison_marker_list = filter_comparison_markers(marker_list)
    nfs_marker_list = filter_nfs_markers(comparison_marker_list)
    print(f"  Ground truth: {len(gt_df):,} pairs")
    print(f"  Marker sets: {marker_list}")
    print(f"  Comparison marker sets: {comparison_marker_list}")
    print(f"  NFS-only marker sets: {nfs_marker_list}")

    # Load results
    print("\n[2] Loading results...")
    all_res = []
    for ms in marker_list:
        merged = merge_results(gt_df, ms, results_dir)
        print(f"  {ms}: {merged['IBS'].notna().sum():,} pairs with data")
        all_res.append(merged)
    all_df = pd.concat(all_res, ignore_index=True)
    all_df.to_csv(eval_dir / "all_results_combined.csv", index=False)

    # ROC
    print("\n[3] Calculating ROC metrics...")
    roc_results = calculate_all_roc_scenarios(all_df, marker_list)
    roc_results.to_csv(eval_dir / "roc_results.csv", index=False)

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
        for metric in ['IBS', 'IBD', 'Kinship']:
            plot_relationship_distribution_single(all_df, ms, metric, DD/f"boxplot_relationship_{metric}_{ms}.png", kind='box')
            plot_relationship_distribution_single(all_df, ms, metric, DD/f"violin_relationship_{metric}_{ms}.png", kind='violin')

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
        ('close_vs_unrelated', lambda d: d['Degree'].isin([1,2,3,4]), lambda d: d['Is_Related']==False, '1-4 vs Unrelated'),
        ('distant_vs_unrelated', lambda d: d['Degree'].isin([5,6]), lambda d: d['Is_Related']==False, '5-6 vs Unrelated'),
        ('4th_vs_unrelated', lambda d: d['Degree']==4, lambda d: d['Is_Related']==False, '4 vs Unrelated'),
        ('5th_vs_unrelated', lambda d: d['Degree']==5, lambda d: d['Is_Related']==False, '5 vs Unrelated'),
        ('6th_vs_unrelated', lambda d: d['Degree']==6, lambda d: d['Is_Related']==False, '6 vs Unrelated'),
    ]
    for name, pos, neg, title in roc_scenarios:
        plot_roc_curves(all_df, name, pos, neg, title, comparison_marker_list, DR/f"roc_{name}.png")
        print(f"    Saved: roc_{name}.png")
        if len(nfs_marker_list) > 1:
            plot_roc_curves(all_df, name, pos, neg, f"{title} (NFS only)", nfs_marker_list, DR/f"roc_{name}_nfs_only.png")
            print(f"    Saved: roc_{name}_nfs_only.png")

    # [9] AUC heatmap + Adjacent discrimination
    print("\n[9] Performance comparison...")
    for m in ['IBS','IBD','Kinship']:
        plot_auc_heatmap(roc_results, m, comparison_marker_list, DC/f"auc_heatmap_{m}.png")
        if len(nfs_marker_list) > 1:
            plot_auc_heatmap(roc_results, m, nfs_marker_list, DC/f"auc_heatmap_{m}_nfs_only.png")
    plot_adjacent_discrimination(roc_results, comparison_marker_list, DC/"adjacent_discrimination.png")
    if len(nfs_marker_list) > 1:
        plot_adjacent_discrimination(roc_results, nfs_marker_list, DC/"adjacent_discrimination_nfs_only.png")
    print(f"    Saved: adjacent_discrimination.png")

    # [10] Scatter
    print("\n[10] Scatter plots...")
    for ms in marker_list:
        plot_scatter_expected_vs_observed(all_df, ms, DS/f"scatter_{ms}.png")

    # [11] NEW: Marker comparison overlay
    print("\n[11] Marker comparison overlay...")
    if len(comparison_marker_list) > 1:
        for m in ['IBS','IBD','Kinship']:
            plot_marker_comparison_overlay(all_df, comparison_marker_list, m, DC/f"marker_overlay_{m}.png")
    if len(nfs_marker_list) > 1:
        for m in ['IBS','IBD','Kinship']:
            plot_marker_comparison_overlay(all_df, nfs_marker_list, m, DC/f"marker_overlay_{m}_nfs_only.png", title_suffix='NFS only')

    pair_specs = [
        (['NFS_12K', 'Kintelligence'], '12K vs Kintelligence', '12k_vs_kintelligence'),
        (['NFS_6K', 'QIAseq'], '6K vs QIAseq', '6k_vs_qiaseq'),
    ]
    for pair_markers, pair_title, pair_tag in pair_specs:
        available = [ms for ms in pair_markers if ms in marker_list]
        if len(available) == 2:
            for m in ['IBS', 'IBD', 'Kinship']:
                plot_marker_comparison_overlay(all_df, available, m, DC/f"marker_overlay_{pair_tag}_{m}.png", title_suffix=pair_title)

    # [12] NEW: Per-degree summary stats
    print("\n[12] Per-degree summary statistics...")
    generate_degree_summary_stats(all_df, comparison_marker_list, reports_dir/"degree_summary_stats.csv", DM/"degree_summary.png")
    if len(nfs_marker_list) > 1:
        generate_degree_summary_stats(all_df, nfs_marker_list, reports_dir/"degree_summary_stats_nfs_only.csv", DM/"degree_summary_nfs_only.png")

    # [13] NEW: Effect size
    print("\n[13] Effect size (Cohen's d)...")
    plot_effect_size_adjacent(all_df, comparison_marker_list, DM/"effect_size_adjacent.png")
    if len(nfs_marker_list) > 1:
        plot_effect_size_adjacent(all_df, nfs_marker_list, DM/"effect_size_adjacent_nfs_only.png")

    # [14] NEW: Confusion matrices
    print("\n[14] Confusion matrices...")
    plot_confusion_matrices(all_df, roc_results, comparison_marker_list, DM/"confusion_matrices.png")
    if len(nfs_marker_list) > 0:
        plot_confusion_matrices(all_df, roc_results, nfs_marker_list, DM/"confusion_matrices_nfs_only.png")

    # [15] Report
    print("\n[15] Generating report...")
    generate_report(all_df, roc_results, marker_list, reports_dir/"kinship_analysis_report.txt")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Related vs Unrelated (AUC)")
    print("=" * 70)
    sdf = roc_results[roc_results['Scenario']=='related_vs_unrelated']
    print(f"\n{'Marker':<15} {'IBS':>10} {'IBD':>10} {'KCs':>10}")
    print("-" * 50)
    for mk in marker_list:
        md = sdf[sdf['Marker_Set']==mk]
        vals = {}
        for m in ['IBS','IBD','Kinship']:
            r = md[md['Metric']==m]['AUC'].values
            vals[m] = f"{r[0]:.4f}" if len(r)>0 and pd.notna(r[0]) else "N/A"
        print(f"{mk:<15} {vals['IBS']:>10} {vals['IBD']:>10} {vals['Kinship']:>10}")
    print(f"\nStep 5 outputs in: {eval_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Step 5: evaluate existing Step 3/4 kinship outputs without rerunning PLINK/KING')
    parser.add_argument('--analysis-dir', '--outdir', dest='analysis_dir', default=str(DEFAULT_ANALYSIS_DIR),
                        help='Directory containing Step 3/4 outputs (default: 06_kinship_analysis)')
    parser.add_argument('--eval-dir', default=f"{DEFAULT_WORK_DIR}/07_evaluate_kinship",
                        help='Directory for Step 5 outputs (default: 07_evaluate_kinship)')
    parser.add_argument('--markers', nargs='+', default=None,
                        help='Optional marker set order/list. If omitted, markers are inferred from *_plink.genome files.')
    args = parser.parse_args()
    args.marker_list = args.markers or []
    return args


def main():
    args = parse_args()
    print("=" * 70)
    print("STEP 5: EVALUATE KINSHIP RESULTS")
    print("=" * 70)
    print(f"  Step 3/4 input: {args.analysis_dir}")
    print(f"  Step 5 output: {args.eval_dir}")
    if args.marker_list:
        print(f"  Marker sets: {args.marker_list}")
    else:
        print("  Marker sets: inferred from Step 3 results")
    step5_evaluate(args)
    print("\nSTEP 5 COMPLETE")


if __name__ == "__main__":
    main()
