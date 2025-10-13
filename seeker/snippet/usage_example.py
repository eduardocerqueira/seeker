#date: 2025-10-13T17:05:53Z
#url: https://api.github.com/gists/2f5044f7fa8e887269ff7f94cb7dfc7e
#owner: https://api.github.com/users/fffclaypool

k = [21, 30, 16]
n = [30, 50, 20]
# 例: 目標比 π=[0.5,0.3,0.2]、観測比 ãπ=[0.3,0.5,0.2] → w=π/ãπ=[1.6667,0.6,1.0]
w = [1.6667, 0.6, 1.0]

pm_w, p_hat_w, Nw, n_eff = weighted_posterior_mean_from_segments(
    k_list=k, n_list=n, w_list=w, prior_mean=0.75, prior_strength=300
)
print(pm_w, p_hat_w, Nw, n_eff)
# pm_w ~ 0.737, p_hat_w ~ 0.69, Nw ~ 100, n_eff ~ 82.4（概ねの目安）
