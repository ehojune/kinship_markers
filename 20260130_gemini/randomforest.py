import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# 경로 설정 (사용자 환경 반영)
BASE_DIR = Path("/BiO/Access/ehojune/kinship/Analysis/20251031_wgrs/06_kinship_analysis_ver0129")
GT_FILE = BASE_DIR / "family_relationships.csv"
RESULT_FILE = BASE_DIR / "all_results_combined_v3.csv" # KING 결과 통합본

# 1. 데이터 로드 및 전처리
df_gt = pd.read_csv(GT_FILE)
df_res = pd.read_csv(RESULT_FILE)

# NFS_36K 마커 데이터만 필터링해서 학습 (가장 성능이 좋으므로)
df_36k = df_res[df_res['Marker_Set'] == 'NFS_36K'].copy()

# Ground Truth와 병합 (샘플 쌍 ID 기준)
# 주의: ID1, ID2 정렬 상태에 따라 병합이 안될 수 있으므로 쌍을 정렬하여 키 생성 필요
df_36k['pair_key'] = df_36k.apply(
    lambda x: "-".join(sorted([str(x['Sample1']), str(x['Sample2'])])), axis=1
)

df_gt['pair_key'] = df_gt.apply(lambda x: "-".join(sorted([str(x['Sample1']), str(x['Sample2'])])), axis=1)

data = pd.merge(df_36k.drop(columns=['Degree']),
                df_gt[['pair_key', 'Degree']],
                on='pair_key')


# 2. 피처 선정 (ML이 학습할 유전적 지표들)
features = ['IBD', 'IBS', 'Z0', 'Z1', 'Z2', 'Kinship']

data = data.dropna(subset=features + ['Degree'])

X = data[features]
y = data['Degree']

# 3. 모델 학습 (Random Forest)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# 4. 결과 출력
y_pred = rf.predict(X_test)
print("=== NFS_36K 기반 친족 관계 분류 성능 리포트 ===")
print(classification_report(y_test, y_pred))

# 5. 피처 중요도 시각화 (어떤 지표가 친족 판별에 가장 중요한가?)
plt.figure(figsize=(10, 6))
importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)
sns.barplot(x=importances, y=importances.index)
plt.title("Feature Importance for Kinship Classification")
plt.savefig(BASE_DIR / "figures/ml_feature_importance.png")