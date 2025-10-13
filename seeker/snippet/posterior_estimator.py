#date: 2025-10-13T17:05:53Z
#url: https://api.github.com/gists/2f5044f7fa8e887269ff7f94cb7dfc7e
#owner: https://api.github.com/users/fffclaypool

from __future__ import annotations
from typing import List, Tuple


def weighted_posterior_mean_from_segments(
    k_list: List[float],
    n_list: List[float],
    w_list: List[float],
    prior_mean: float = 0.75,
    prior_strength: float = 300.0,
    clip_min: float | None = 0.3,
    clip_max: float | None = 3.0,
) -> Tuple[float, float, float, float]:
    """
    分布補正（重み付き）＋ Kish の n_eff 近似でベイズ事後平均を返す。

    引数:
        k_list: 各セグメントの「成功件数」リスト [k_s]。例: 正解ラベル数。各要素は 0 以上で、対応する n_list[i] 以下。
        n_list: 各セグメントの「総件数」リスト [n_s]。= 成功 + 失敗。各要素は 0 以上。
        w_list: 各セグメントの「重み」リスト [w_s]。通常は w_s = 目標比 π_s / 観測比 ˆπ_s を（必要ならクリップ後）渡す。
                層別の不均等サンプリングをした場合は「設計重み × 校正重み」を渡す。
        prior_mean: 事前平均 m0（0〜1）。例: 直近30日やSTG回帰セットの分布補正後の平均精度。
        prior_strength: 事前の強さ n0（>0）。過去情報の仮想サンプル数。速報で過去にどれだけ寄せるかで調整。
        clip_min: 重みの下限。極端な重みによる分散悪化を防ぐためのクリップ（None で無効）。
        clip_max: 重みの上限。極端な重みによる分散悪化を防ぐためのクリップ（None で無効）。

    返り値:
        posterior_mean : ベイズ事後平均（事前 Beta を加味した速報値）
        p_hat_w        : 重み付き精度 = (Σ w_s k_s) / (Σ w_s n_s)
        N_w            : 重み付き総件数 = Σ w_s n_s（見かけの件数）
        n_eff          : 有効サンプル数（Kish）= (N_w^2) / Σ(w_s^2 n_s)（実質の情報量）

    注意:
      - w_list は必要に応じて clip_min/clip_max でクリップ。
      - n_eff はセグメント近似: Σ(w^2 * n) を使用（イベント粒度の w が無い想定）。
      - prior_mean / prior_strength は月次などで棚卸しすると良い。
    """

    if not (len(k_list) == len(n_list) == len(w_list)):
        raise ValueError("k_list, n_list, w_list の長さを揃えてください。")
    if any(n < 0 for n in n_list) or any(k < 0 for k in k_list):
        raise ValueError("k_list, n_list は0以上にしてください。")
    if any(k > n for k, n in zip(k_list, n_list)):
        raise ValueError("各セグメントで k <= n を満たしてください。")
    if not (0.0 < prior_mean < 1.0):
        raise ValueError("prior_mean は (0,1) にしてください。")
    if prior_strength <= 0:
        raise ValueError("prior_strength は正にしてください。")

    # 重みのクリップ（推奨）
    if clip_min is not None and clip_max is not None and clip_min > 0 and clip_max >= clip_min:
        w = [min(clip_max, max(clip_min, wi)) for wi in w_list]
    else:
        w = list(w_list)

    # 重み付き成功・件数
    Sw = sum(wi * ki for wi, ki in zip(w, k_list))
    Nw = sum(wi * ni for wi, ni in zip(w, n_list))
    if Nw == 0:
        raise ValueError("重み付き総件数 N_w が0です。n_list / w_list を確認してください。")

    p_hat_w = Sw / Nw

    # Kish の有効サンプル数（セグメント近似）
    sum_w2_n = sum((wi ** 2) * ni for wi, ni in zip(w, n_list))
    if sum_w2_n == 0:
        raise ValueError("sum(w^2 * n) が0です。n_list / w_list を確認してください。")
    n_eff = (Nw ** 2) / sum_w2_n

    # 等価カウントに写像
    S_eff = p_hat_w * n_eff
    F_eff = (1.0 - p_hat_w) * n_eff

    # 事前 → 事後
    alpha0 = prior_mean * prior_strength
    beta0 = (1.0 - prior_mean) * prior_strength
    posterior_mean = (alpha0 + S_eff) / (alpha0 + beta0 + S_eff + F_eff)

    return posterior_mean, p_hat_w, Nw, n_eff
