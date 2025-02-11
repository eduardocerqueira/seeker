#date: 2025-02-11T16:58:40Z
#url: https://api.github.com/gists/72f5143d27d6af35dc1ecb0a56aafd1a
#owner: https://api.github.com/users/kakira9618

import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.special import gamma

def generalized_normal_pdf(x, beta, mu=0):
    """
    分散を1に標準化した一般化正規分布の確率密度関数を返す。
    パラメータ:
      beta : 形状パラメータ
      mu   : 平均（ここでは 0 とする）
    """
    # 分散1にするためのα
    alpha = np.sqrt(gamma(1/beta) / gamma(3/beta))
    coeff = beta / (2 * alpha * gamma(1/beta))
    return coeff * np.exp(- (np.abs(x-mu) / alpha)**beta)

def excess_kurtosis(beta):
    """
    一般化正規分布の理論上の尖度（第四中心モーメント／分散の2乗）から余尖度を返す。
    """
    kurt = (gamma(5/beta) * gamma(1/beta)) / (gamma(3/beta)**2)
    return kurt - 3

# 描画用のx軸の範囲（重たい裾の場合を考慮して広めに）
x = np.linspace(-5, 5, 1000)

# 比較する β の値：β=1 (leptokurtic), β=2 (正規分布), β=4 (platykurtic)
beta_values = [1, 2, 4]
colors = ['blue', 'green', 'red']

plt.figure(figsize=(8, 5))
for beta, color in zip(beta_values, colors):
    pdf = generalized_normal_pdf(x, beta)
    exc_kurt = excess_kurtosis(beta)
    label = f"β = {beta}, excess kurtosis = {exc_kurt:.2f}"
    plt.plot(x, pdf, label=label, color=color, lw=2)

plt.title("一般化正規分布の確率密度 (分散=1)")
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.legend()
plt.grid(True)
plt.show()

