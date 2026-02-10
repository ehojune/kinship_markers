#!/usr/bin/env python3
"""
Step 5: Comprehensive Kinship Analysis Evaluation (FIXED VERSION)

Key fixes:
1. Include ALL pairs (within-family + between-family) in plots
2. Add relationship type plots (Parent, Sibling, Uncle, Cousin, etc.)
3. Fix heatmap colors (standard RdYlBu)
4. Proper degree labeling

Run after: 04_generate_ground_truth_fixed.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
HOME = Path.home()
WORK_DIR = HOME / "kinship/Analysis/20251031_wgrs/06_kinship_analysis"
RESULTS_DIR = WORK_DIR / "results"
FIGURES_DIR = WORK_DIR / "figures"
REPORTS_DIR = WORK_DIR / "reports"

for d in [FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Subdirectories
FIG_HEATMAP_DIR = FIGURES_DIR / "heatmaps"
FIG_DIST_DIR = FIGURES_DIR / "distributions"
FIG_ROC_DIR = FIGURES_DIR / "roc_curves"
FIG_SCATTER_DIR = FIGURES_DIR / "scatter"
FIG_COMPARISON_DIR = FIGURES_DIR / "comparison"

for d in [FIG_HEATMAP_DIR, FIG_DIST_DIR, FIG_ROC_DIR, FIG_SCATTER_DIR, FIG_COMPARISON_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GROUND_TRUTH = WORK_DIR / "family_relationships.csv"

# Marker sets
MARKER_SETS = ['NFS_36K', 'NFS_24K', 'NFS_20K', 'NFS_12K', 'NFS_6K', 'Kintelligence', 'QIAseq']

# Plotting style


plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.facecolor'] = 'white'

# Color schemes
MARKER_COLORS = {
    'NFS_36K': '#1a5276',
    'NFS_24K': '#2874a6',
    'NFS_20K': '#3498db',
    'NFS_12K': '#e74c3c',
    'NFS_6K': '#9b59b6',
    'Kintelligence': '#27ae60',
    'QIAseq': '#f39c12'
}

# Degree colors (for blood relatives)
DEGREE_COLORS = {
    0: '#95a5a6',   # Unrelated - Gray
    1: '#c0392b',   # 1st degree - Dark red
    2: '#e74c3c',   # 2nd degree - Red
    3: '#e67e22',   # 3rd degree - Orange
    4: '#f1c40f',   # 4th degree - Yellow
    5: '#2ecc71',   # 5th degree - Green
    6: '#3498db',   # 6th degree - Blue
    7: '#9b59b6',   # 7th degree - Purple
}

# Relationship type colors
RELATIONSHIP_COLORS = {
    'Parent-Child': '#c0392b',
    'Sibling': '#e74c3c',
    'Grandparent-Grandchild': '#d35400',
    'Uncle-Nephew': '#e67e22',
    'Cousin': '#f39c12',
    'Grand-Uncle-Nephew': '#27ae60',
    'Cousin-Once-Removed': '#2ecc71',
    'Second-Cousin': '#3498db',
    'Spouse': '#7f8c8d',
    'Unrelated': '#95a5a6',
}

# Relationship order for plotting
RELATIONSHIP_ORDER = [
    'Parent-Child',
    'Sibling', 
    'Grandparent-Grandchild',
    'Uncle-Nephew',
    'Cousin',
    'Grand-Uncle-Nephew',
    'Cousin-Once-Removed',
    'Second-Cousin',
    'Spouse',
    'Unrelated'
]

RELATIONSHIP_LABELS = {
    'Parent-Child': 'Parent-Child\n(1촌)',
    'Sibling': 'Sibling\n(2촌)',
    'Grandparent-Grandchild': 'Grandparent\n(2촌)',
    'Uncle-Nephew': 'Uncle-Nephew\n(3촌)',
    'Cousin': 'Cousin\n(4촌)',
    'Grand-Uncle-Nephew': 'Grand-Uncle\n(5촌)',
    'Cousin-Once-Removed': 'Cousin-1R\n(5촌)',
    'Second-Cousin': '2nd-Cousin\n(6촌)',
    'Spouse': 'Spouse\n(0촌)',
    'Unrelated': 'Unrelated\n(0촌)'
}


# ============================================================
# Data Loading
# ============================================================

def load_plink_genome(filepath):
    """Load PLINK .genome file"""
    if not filepath.exists():
        return None
    df = pd.read_csv(filepath, delim_whitespace=True)
    df['Sample1'] = df['IID1'].astype(str)
    df['Sample2'] = df['IID2'].astype(str)
    df['pair'] = df.apply(lambda r: tuple(sorted([r['Sample1'], r['Sample2']])), axis=1)
    return df[['pair', 'Sample1', 'Sample2', 'PI_HAT', 'DST', 'Z0', 'Z1', 'Z2']]


def load_king_kinship(filepath):
    """Load KING kinship file"""
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


def merge_results(ground_truth, marker_set):
    """Merge PLINK and KING results with ground truth"""
    plink_file = RESULTS_DIR / f"{marker_set}_plink.genome"
    king_file = RESULTS_DIR / f"{marker_set}_king.kin0"
    
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
# 1. Distribution Plots - By Degree (ALL DATA)
# ============================================================

def plot_boxplot_by_degree_all(all_df, marker_set, output_path):
    """
    Boxplot by degree - includes ALL pairs (within + between family)
    """
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    
    if len(df) == 0:
        return
    
    # Create degree label
    def get_degree_label(row):
        if row['Same_Family']:
            if row['Degree'] == 0:
                return 'Spouse/InLaw\n(0촌)'
            else:
                return f"{row['Degree']}촌"
        else:
            return 'Between-Fam\n(Unrel)'
    
    df['Degree_Label'] = df.apply(get_degree_label, axis=1)
    
    # Order
    order = ['1촌', '2촌', '3촌', '4촌', '5촌', '6촌', 'Spouse/InLaw\n(0촌)', 'Between-Fam\n(Unrel)']
    available_order = [o for o in order if o in df['Degree_Label'].values]
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    metrics = ['IBS', 'IBD', 'Kinship']
    
    # Color palette
    palette = []
    for o in available_order:
        if '1촌' in o:
            palette.append(DEGREE_COLORS[1])
        elif '2촌' in o:
            palette.append(DEGREE_COLORS[2])
        elif '3촌' in o:
            palette.append(DEGREE_COLORS[3])
        elif '4촌' in o:
            palette.append(DEGREE_COLORS[4])
        elif '5촌' in o:
            palette.append(DEGREE_COLORS[5])
        elif '6촌' in o:
            palette.append(DEGREE_COLORS[6])
        else:
            palette.append(DEGREE_COLORS[0])
    
    for ax, metric in zip(axes, metrics):
        data = df.dropna(subset=[metric])
        if len(data) == 0:
            ax.set_title(f'{metric} (No Data)')
            continue
        
        # Boxplot
        sns.boxplot(data=data, x='Degree_Label', y=metric, order=available_order,
                   palette=palette, ax=ax, width=0.6, linewidth=1.5,
                   flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.3})
        
        # Add strip plot for individual points
        sns.stripplot(data=data, x='Degree_Label', y=metric, order=available_order,
                     color='black', size=2, alpha=0.3, ax=ax, jitter=True)
        
        # Sample counts
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


def plot_boxplot_by_relationship(all_df, marker_set, output_path):
    """
    Boxplot by relationship type (Parent, Sibling, Uncle, Cousin, etc.)
    """
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    
    if len(df) == 0:
        return
    
    # Filter to relationships we care about
    rel_order = [r for r in RELATIONSHIP_ORDER if r in df['Relationship'].values]
    
    if len(rel_order) == 0:
        return
    
    # Create display labels
    df['Rel_Label'] = df['Relationship'].map(lambda x: RELATIONSHIP_LABELS.get(x, x))
    label_order = [RELATIONSHIP_LABELS.get(r, r) for r in rel_order]
    
    # Color palette
    palette = [RELATIONSHIP_COLORS.get(r, '#95a5a6') for r in rel_order]
    
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    metrics = ['IBS', 'IBD', 'Kinship']
    
    for ax, metric in zip(axes, metrics):
        data = df.dropna(subset=[metric])
        if len(data) == 0:
            continue
        
        sns.boxplot(data=data, x='Rel_Label', y=metric, order=label_order,
                   palette=palette, ax=ax, width=0.6, linewidth=1.5)
        
        sns.stripplot(data=data, x='Rel_Label', y=metric, order=label_order,
                     color='black', size=2, alpha=0.3, ax=ax, jitter=True)
        
        # Sample counts
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


def plot_violin_by_relationship(all_df, marker_set, output_path):
    """
    Violin plot by relationship type
    """
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
    metrics = ['IBS', 'IBD', 'Kinship']
    
    for ax, metric in zip(axes, metrics):
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
# 2. Heatmaps - Standard Colors
# ============================================================

def plot_heatmap_standard(all_df, marker_set, metric, output_path):
    """
    Heatmap with standard seaborn colors (RdYlBu_r)
    """
    df = all_df[all_df['Marker_Set'] == marker_set].copy()
    
    if len(df) == 0 or df[metric].isna().all():
        return
    
    # Get samples (exclude GP* founders for cleaner plot)
    samples = sorted(set(df['Sample1'].tolist() + df['Sample2'].tolist()))
    samples = [s for s in samples if 'GP' not in s and 'GM' not in s]
    n = len(samples)
    
    if n == 0:
        return
    
    # Create matrix
    matrix = np.full((n, n), np.nan)
    sample_idx = {s: i for i, s in enumerate(samples)}
    
    for _, row in df.iterrows():
        s1, s2 = str(row['Sample1']), str(row['Sample2'])
        i = sample_idx.get(s1)
        j = sample_idx.get(s2)
        val = row[metric]
        if i is not None and j is not None and pd.notna(val):
            matrix[i, j] = val
            matrix[j, i] = val
    
    # Fill diagonal
    if metric == 'Kinship':
        np.fill_diagonal(matrix, 0.5)
    else:
        np.fill_diagonal(matrix, 1.0)
    
    # Plot
    figsize = max(14, n * 0.25)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.9))
    
    # Standard colormap
    if metric == 'IBS':
        vmin, vmax = 0.55, 0.85
        cmap = 'RdYlBu_r'
    elif metric == 'IBD':
        vmin, vmax = 0, 0.6
        cmap = 'RdYlBu_r'
    else:  # Kinship
        vmin, vmax = -0.05, 0.3
        cmap = 'RdYlBu_r'
    
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    
    sns.heatmap(matrix, mask=mask, cmap=cmap, vmin=vmin, vmax=vmax,
                square=True, linewidths=0.2, linecolor='white',
                cbar_kws={'shrink': 0.6, 'label': metric},
                ax=ax)
    
    # Labels (shortened)
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


def plot_heatmap_within_family(all_df, marker_set, metric, family, output_path):
    """Heatmap for a single family"""
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
        i = sample_idx.get(s1)
        j = sample_idx.get(s2)
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
                cbar_kws={'shrink': 0.7, 'label': metric},
                ax=ax)
    
    sample_labels = [s.split('-')[-1] for s in samples]
    ax.set_xticklabels(sample_labels, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(sample_labels, rotation=0, fontsize=11)
    ax.set_title(f'Family {family} - {marker_set} - {metric}', fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# ============================================================
# 3. ROC Curves
# ============================================================

def calculate_roc_metrics(y_true, y_score):
    """Calculate ROC metrics"""
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


def calculate_all_roc_scenarios(all_df):
    """Calculate ROC for multiple classification scenarios"""
    
    scenarios = {
        'related_vs_unrelated': {'pos': lambda d: d['Is_Related'] == True, 
                                  'neg': lambda d: d['Is_Related'] == False,
                                  'desc': 'Related vs Unrelated (All)'},
        'blood_within_vs_unrelated': {'pos': lambda d: (d['Same_Family'] == True) & (d['Degree'] > 0),
                                       'neg': lambda d: d['Is_Related'] == False,
                                       'desc': 'Blood Relatives vs Unrelated'},
        'close_vs_unrelated': {'pos': lambda d: d['Degree'].isin([1,2,3,4]),
                               'neg': lambda d: d['Is_Related'] == False,
                               'desc': '1-4촌 vs Unrelated'},
        'distant_vs_unrelated': {'pos': lambda d: d['Degree'].isin([5,6]),
                                  'neg': lambda d: d['Is_Related'] == False,
                                  'desc': '5-6촌 vs Unrelated'},
        '1st_vs_2nd': {'pos': lambda d: d['Degree'] == 1,
                       'neg': lambda d: d['Degree'] == 2,
                       'desc': '1촌 vs 2촌'},
        '2nd_vs_3rd': {'pos': lambda d: d['Degree'] == 2,
                       'neg': lambda d: d['Degree'] == 3,
                       'desc': '2촌 vs 3촌'},
        '3rd_vs_4th': {'pos': lambda d: d['Degree'] == 3,
                       'neg': lambda d: d['Degree'] == 4,
                       'desc': '3촌 vs 4촌'},
        '4th_vs_5th': {'pos': lambda d: d['Degree'] == 4,
                       'neg': lambda d: d['Degree'] == 5,
                       'desc': '4촌 vs 5촌'},
        '5th_vs_6th': {'pos': lambda d: d['Degree'] == 5,
                       'neg': lambda d: d['Degree'] == 6,
                       'desc': '5촌 vs 6촌'},
        '4th_vs_unrelated': {'pos': lambda d: d['Degree'] == 4,
                             'neg': lambda d: d['Is_Related'] == False,
                             'desc': '4촌 vs Unrelated'},
        '5th_vs_unrelated': {'pos': lambda d: d['Degree'] == 5,
                             'neg': lambda d: d['Is_Related'] == False,
                             'desc': '5촌 vs Unrelated'},
        '6th_vs_unrelated': {'pos': lambda d: d['Degree'] == 6,
                             'neg': lambda d: d['Is_Related'] == False,
                             'desc': '6촌 vs Unrelated'},
    }
    
    results = []
    
    for marker_set in MARKER_SETS:
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


def plot_roc_curves(all_df, scenario_name, pos_filter, neg_filter, title, output_path):
    """Plot ROC curves for all marker sets"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ['IBS', 'IBD', 'Kinship']
    
    for ax, metric in zip(axes, metrics):
        for marker_set in MARKER_SETS:
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
# 4. Performance Comparison Charts
# ============================================================

def plot_auc_heatmap(roc_results, metric, output_path):
    """Heatmap: Marker Set x Scenario AUC"""
    data = roc_results[roc_results['Metric'] == metric].copy()
    
    if len(data) == 0:
        return
    
    pivot = data.pivot(index='Marker_Set', columns='Scenario', values='AUC')
    
    # Reorder
    marker_order = [m for m in MARKER_SETS if m in pivot.index]
    pivot = pivot.reindex(marker_order)
    
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


def plot_adjacent_discrimination(roc_results, output_path):
    """Adjacent degree discrimination line plot"""
    adjacent = ['1st_vs_2nd', '2nd_vs_3rd', '3rd_vs_4th', '4th_vs_5th', '5th_vs_6th']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    metrics = ['IBS', 'IBD', 'Kinship']
    
    for ax, metric in zip(axes, metrics):
        metric_data = roc_results[(roc_results['Metric'] == metric) & 
                                  (roc_results['Scenario'].isin(adjacent))]
        
        for marker_set in MARKER_SETS:
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
# 5. Scatter Plots
# ============================================================

def plot_scatter_expected_vs_observed(all_df, marker_set, output_path):
    """Scatter: Expected vs Observed kinship"""
    df = all_df[(all_df['Marker_Set'] == marker_set)].copy()
    
    if len(df) == 0:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, metric in zip(axes, ['IBS', 'IBD', 'Kinship']):
        data = df.dropna(subset=[metric])
        
        # Color by degree
        for deg in sorted(data['Degree'].unique()):
            deg_data = data[data['Degree'] == deg]
            color = DEGREE_COLORS.get(deg, '#95a5a6')
            label = f"{deg}촌" if deg > 0 else "Unrel"
            ax.scatter(deg_data['Expected_Kinship'], deg_data[metric],
                      c=color, label=label, alpha=0.6, s=30, edgecolors='white', linewidth=0.3)
        
        # Correlation
        valid = data[['Expected_Kinship', metric]].dropna()
        if len(valid) > 2:
            corr = valid['Expected_Kinship'].corr(valid[metric])
            ax.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=12, fontweight='bold')
        
        if metric == 'Kinship':
            lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])]
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
# 6. Report Generation
# ============================================================

def generate_report(all_df, roc_results):
    """Generate comprehensive text report"""
    
    report_path = REPORTS_DIR / "kinship_analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("KINSHIP MARKER PERFORMANCE EVALUATION REPORT\n")
        f.write("=" * 100 + "\n\n")
        
        # Dataset summary
        f.write("1. DATASET SUMMARY\n")
        f.write("-" * 50 + "\n")
        
        sample_df = all_df[all_df['Marker_Set'] == MARKER_SETS[0]]
        n_pairs = len(sample_df)
        n_related = len(sample_df[sample_df['Is_Related'] == True])
        n_within_fam = len(sample_df[sample_df['Same_Family'] == True])
        
        f.write(f"  Total pairs: {n_pairs:,}\n")
        f.write(f"  Blood-related pairs: {n_related:,}\n")
        f.write(f"  Within-family pairs: {n_within_fam:,}\n")
        f.write(f"  Between-family pairs: {n_pairs - n_within_fam:,}\n\n")
        
        # By relationship
        f.write("2. RELATIONSHIP DISTRIBUTION\n")
        f.write("-" * 50 + "\n")
        
        rel_counts = sample_df['Relationship'].value_counts()
        for rel, count in rel_counts.items():
            f.write(f"  {rel:<30}: {count:>6}\n")
        f.write("\n")
        
        # By degree
        f.write("3. DEGREE DISTRIBUTION\n")
        f.write("-" * 50 + "\n")
        
        deg_counts = sample_df.groupby('Degree').agg({
            'Sample1': 'count',
            'Expected_Kinship': 'first'
        })
        
        for deg, row in deg_counts.iterrows():
            label = f"{deg}촌" if deg > 0 else "Unrelated"
            f.write(f"  {label:<15}: {int(row['Sample1']):>6} pairs  (φ = {row['Expected_Kinship']:.4f})\n")
        f.write("\n")
        
        # Performance summary
        f.write("4. CLASSIFICATION PERFORMANCE (AUC)\n")
        f.write("-" * 80 + "\n")
        
        # Related vs Unrelated
        f.write("\n  [Related vs Unrelated]\n")
        f.write(f"  {'Marker':<15} {'IBS':>10} {'IBD':>10} {'Kinship':>10}\n")
        f.write("  " + "-" * 50 + "\n")
        
        scenario_df = roc_results[roc_results['Scenario'] == 'related_vs_unrelated']
        for marker in MARKER_SETS:
            marker_data = scenario_df[scenario_df['Marker_Set'] == marker]
            ibs = marker_data[marker_data['Metric'] == 'IBS']['AUC'].values
            ibd = marker_data[marker_data['Metric'] == 'IBD']['AUC'].values
            kin = marker_data[marker_data['Metric'] == 'Kinship']['AUC'].values
            
            ibs_str = f"{ibs[0]:.4f}" if len(ibs) > 0 and pd.notna(ibs[0]) else "N/A"
            ibd_str = f"{ibd[0]:.4f}" if len(ibd) > 0 and pd.notna(ibd[0]) else "N/A"
            kin_str = f"{kin[0]:.4f}" if len(kin) > 0 and pd.notna(kin[0]) else "N/A"
            
            f.write(f"  {marker:<15} {ibs_str:>10} {ibd_str:>10} {kin_str:>10}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("END OF REPORT\n")
    
    print(f"  Report: {report_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 80)
    print("COMPREHENSIVE KINSHIP ANALYSIS EVALUATION (FIXED)")
    print("=" * 80)
    
    # Load ground truth
    print("\n[1] Loading ground truth...")
    if not GROUND_TRUTH.exists():
        print(f"  ERROR: {GROUND_TRUTH} not found!")
        print("  Please run 04_generate_ground_truth_fixed.py first!")
        return
    
    ground_truth = pd.read_csv(GROUND_TRUTH)
    print(f"  Total pairs: {len(ground_truth):,}")
    print(f"  Relationships: {ground_truth['Relationship'].nunique()}")
    
    # Load all results
    print("\n[2] Loading results for all marker sets...")
    all_results = []
    
    for marker_set in MARKER_SETS:
        print(f"  {marker_set}...", end=" ")
        merged = merge_results(ground_truth, marker_set)
        all_results.append(merged)
        n_valid = merged['IBS'].notna().sum()
        print(f"{n_valid:,} pairs with data")
    
    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(WORK_DIR / "all_results_combined.csv", index=False)
    
    # Calculate ROC
    print("\n[3] Calculating ROC metrics...")
    roc_results = calculate_all_roc_scenarios(all_df)
    roc_results.to_csv(WORK_DIR / "roc_results.csv", index=False)
    
    # ==========================================
    # Generate Figures
    # ==========================================
    print("\n" + "=" * 80)
    print("GENERATING FIGURES")
    print("=" * 80)
    
    # 1. Distribution by Degree (ALL pairs)
    print("\n[4] Distribution plots by DEGREE...")
    for marker_set in MARKER_SETS:
        output = FIG_DIST_DIR / f"boxplot_degree_{marker_set}.png"
        plot_boxplot_by_degree_all(all_df, marker_set, output)
    
    # 2. Distribution by Relationship Type
    print("\n[5] Distribution plots by RELATIONSHIP TYPE...")
    for marker_set in MARKER_SETS:
        output = FIG_DIST_DIR / f"boxplot_relationship_{marker_set}.png"
        plot_boxplot_by_relationship(all_df, marker_set, output)
        
        output = FIG_DIST_DIR / f"violin_relationship_{marker_set}.png"
        plot_violin_by_relationship(all_df, marker_set, output)
    
    # 3. Heatmaps (standard colors)
    print("\n[6] Heatmaps...")
    for marker_set in MARKER_SETS:
        for metric in ['IBS', 'IBD', 'Kinship']:
            output = FIG_HEATMAP_DIR / f"heatmap_{marker_set}_{metric}.png"
            plot_heatmap_standard(all_df, marker_set, metric, output)
    
    # 4. Per-family heatmaps
    print("\n[7] Per-family heatmaps (NFS_36K)...")
    families = all_df['Family1'].unique()
    for fam in sorted(families)[:5]:
        for metric in ['IBS', 'Kinship']:
            output = FIG_HEATMAP_DIR / f"heatmap_family{fam}_{metric}.png"
            plot_heatmap_within_family(all_df, 'NFS_36K', metric, fam, output)
    
    # 5. ROC curves
    print("\n[8] ROC curves...")
    roc_scenarios = [
        ('related_vs_unrelated', lambda d: d['Is_Related'] == True, lambda d: d['Is_Related'] == False, 'Related vs Unrelated'),
        ('close_vs_unrelated', lambda d: d['Degree'].isin([1,2,3,4]), lambda d: d['Is_Related'] == False, '1-4촌 vs Unrelated'),
        ('distant_vs_unrelated', lambda d: d['Degree'].isin([5,6]), lambda d: d['Is_Related'] == False, '5-6촌 vs Unrelated'),
        ('4th_vs_unrelated', lambda d: d['Degree'] == 4, lambda d: d['Is_Related'] == False, '4촌 vs Unrelated'),
        ('6th_vs_unrelated', lambda d: d['Degree'] == 6, lambda d: d['Is_Related'] == False, '6촌 vs Unrelated'),
    ]
    
    for name, pos, neg, title in roc_scenarios:
        output = FIG_ROC_DIR / f"roc_{name}.png"
        plot_roc_curves(all_df, name, pos, neg, title, output)
        print(f"    Saved: roc_{name}.png")
    
    # 6. Performance comparison
    print("\n[9] Performance comparison charts...")
    for metric in ['IBS', 'IBD', 'Kinship']:
        output = FIG_COMPARISON_DIR / f"auc_heatmap_{metric}.png"
        plot_auc_heatmap(roc_results, metric, output)
    
    output = FIG_COMPARISON_DIR / "adjacent_discrimination.png"
    plot_adjacent_discrimination(roc_results, output)
    print(f"    Saved: adjacent_discrimination.png")
    
    # 7. Scatter plots
    print("\n[10] Scatter plots...")
    for marker_set in MARKER_SETS[:3]:
        output = FIG_SCATTER_DIR / f"scatter_{marker_set}.png"
        plot_scatter_expected_vs_observed(all_df, marker_set, output)
    
    # 8. Report
    print("\n[11] Generating report...")
    generate_report(all_df, roc_results)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Related vs Unrelated (AUC)")
    print("=" * 80)
    
    scenario_df = roc_results[roc_results['Scenario'] == 'related_vs_unrelated']
    print(f"\n{'Marker':<15} {'IBS':>10} {'IBD':>10} {'Kinship':>10}")
    print("-" * 50)
    
    for marker in MARKER_SETS:
        marker_data = scenario_df[scenario_df['Marker_Set'] == marker]
        ibs = marker_data[marker_data['Metric'] == 'IBS']['AUC'].values
        ibd = marker_data[marker_data['Metric'] == 'IBD']['AUC'].values
        kin = marker_data[marker_data['Metric'] == 'Kinship']['AUC'].values
        
        ibs_str = f"{ibs[0]:.4f}" if len(ibs) > 0 and pd.notna(ibs[0]) else "N/A"
        ibd_str = f"{ibd[0]:.4f}" if len(ibd) > 0 and pd.notna(ibd[0]) else "N/A"
        kin_str = f"{kin[0]:.4f}" if len(kin) > 0 and pd.notna(kin[0]) else "N/A"
        
        print(f"{marker:<15} {ibs_str:>10} {ibd_str:>10} {kin_str:>10}")
    
    print(f"\nOutput: {WORK_DIR}")


if __name__ == "__main__":
    main()