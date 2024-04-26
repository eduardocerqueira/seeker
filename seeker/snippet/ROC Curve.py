#date: 2024-04-26T17:01:02Z
#url: https://api.github.com/gists/a44f3dc71dbf55730620c6af73f0bd14
#owner: https://api.github.com/users/Sugtoku

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ダミーデータの生成
# 特徴量：株価の動き（3か月ごとの変動率）、ターゲット：売上高増減（0: 減少、1: 増加）
np.random.seed(0)
data_size = 500
features = np.random.normal(0, 1, (data_size, 10))  # 10個の特徴量
target_profit_increase = np.where(np.random.rand(data_size) > 0.5, 1, 0)  # 50%の確率で増益

# DataFrameに変換
df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(features.shape[1])])
df['Profit_Increase'] = target_profit_increase

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['Profit_Increase'], test_size=0.2, random_state=42)

# LightGBMの設定
train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbose': -1,
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31
}

# モデルの訓練
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data], early_stopping_rounds=10)

# テストデータに対する予測（確率）
y_pred_prob = bst.predict(X_test)

# ROC curveとAUCの計算
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# ROC curveの描画
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
