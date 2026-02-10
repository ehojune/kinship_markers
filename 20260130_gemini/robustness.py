import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 경로 설정
BASE_DIR = Path("/BiO/Access/ehojune/kinship/Analysis/20251031_wgrs/06_kinship_analysis_ver0129")
ROC_FILE = BASE_DIR / "roc_results_all_scenarios_v3.csv"

# 1. 데이터 로드
df_roc = pd.read_csv(ROC_FILE)

# 2. 마커 수 정보 매핑 (ls 결과 참고)
marker_counts = {
    'NFS_36K': 36072, 'NFS_24K': 24400, 'NFS_20K': 19849, 
    'NFS_12K': 12259, 'NFS_6K': 6431,
    'Kintelligence': 9865, 'QIAseq': 5489
}
df_roc['Marker_Count'] = df_roc['Marker_Set'].map(marker_counts)

# 3. 6촌 판별(가장 어려운 태스크) 시나리오만 선택
hard_task = df_roc[df_roc['Scenario'] == '6th_vs_unrelated']

# 4. 시각화: 마커 개수(X) vs 정확도(AUC)
plt.figure(figsize=(12, 7))
sns.lineplot(data=hard_task, x='Marker_Count', y='AUC', marker='o', hue='Metric', linewidth=2.5)

# 타사 제품 성능을 점선/별표로 표시하여 가독성 높임
others = hard_task[hard_task['Marker_Set'].isin(['Kintelligence', 'QIAseq'])]
for i, row in others.iterrows():
    plt.annotate(row['Marker_Set'], (row['Marker_Count'], row['AUC']), 
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')

plt.title("Robustness Analysis: Performance vs. Number of Markers (6th Degree Detection)")
plt.xlabel("Number of Markers")
plt.ylabel("AUC (Accuracy)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig(BASE_DIR / "figures/robustness_comparison.png")
print(df_roc['Scenario'].unique())