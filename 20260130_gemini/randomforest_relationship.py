import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 1. 경로 설정 및 데이터 로드
BASE_DIR = Path("/BiO/Access/ehojune/kinship/Analysis/20251031_wgrs/06_kinship_analysis_ver0129")
GT_FILE = BASE_DIR / "family_relationships.csv"
RESULT_FILE = BASE_DIR / "all_results_combined_v3.csv"

df_gt = pd.read_csv(GT_FILE)
df_res = pd.read_csv(RESULT_FILE)

# NFS_36K 마커 데이터 필터링
df_36k = df_res[df_res['Marker_Set'] == 'NFS_36K'].copy()

# 2. 데이터 병합 (Pair Key 생성)
# KING 결과와 Ground Truth의 샘플 순서가 다를 수 있으므로 정렬하여 매핑
df_36k['pair_key'] = df_36k.apply(lambda x: "-".join(sorted([str(x['Sample1']), str(x['Sample2'])])), axis=1)
df_gt['pair_key'] = df_gt.apply(lambda x: "-".join(sorted([str(x['Sample1']), str(x['Sample2'])])), axis=1)

# Relationship 컬럼을 포함하여 병합
data = pd.merge(
    df_36k.drop(columns=['Degree'], errors='ignore'), 
    df_gt[['pair_key', 'Relationship']], 
    on='pair_key'
)

# 3. 피처 및 타겟 설정
# 관계 구분에 핵심적인 IBD 지표들 (Z0, Z1, Z2) 포함
features = ['IBD', 'IBS', 'Z0', 'Z1', 'Z2', 'Kinship']
data = pd.merge(
    df_36k.drop(columns=['Degree', 'Relationship'], errors='ignore'),
    df_gt[['pair_key', 'Relationship']],
    on='pair_key'
)

X = data[features]
y = data['Relationship']

# 4. 모델 학습 (Stratified Split으로 관계별 비율 유지)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

# 5. 결과 리포트 출력
y_pred = rf.predict(X_test)
print("=== NFS_36K 기반 관계 유형(Relationship) 분류 리포트 ===")
print(classification_report(y_test, y_pred))

# 6. 시각화: 혼동 행렬 (Confusion Matrix)
plt.figure(figsize=(12, 10))
cmd = ConfusionMatrixDisplay.from_estimator(
    rf, X_test, y_test, 
    display_labels=rf.classes_,
    xticks_rotation=45,
    cmap='Blues',
    normalize='true' # 비율로 표시
)
plt.title("Confusion Matrix: Relationship Classification")
plt.tight_layout()
plt.savefig(BASE_DIR / "figures/ml_relationship_confusion_matrix.png")

# 7. 피처 중요도 확인
plt.figure(figsize=(10, 6))
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=importances, y=importances.index, palette='viridis')
plt.title("Feature Importance: Which metric defines the relationship?")
plt.savefig(BASE_DIR / "figures/ml_rel_feature_importance.png")

plt.show()