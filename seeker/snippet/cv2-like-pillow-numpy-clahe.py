#date: 2024-10-02T16:39:29Z
#url: https://api.github.com/gists/3aa39990d30c2cc70fdcb4131400017f
#owner: https://api.github.com/users/nepia11

import numpy as np
from PIL import Image


def clahe_optimized(image, clip_limit=40.0, grid_size=(8, 8)):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    if image.dtype != np.uint8:
        raise ValueError("Only 8-bit images are supported")

    h, w = image.shape
    grid_h, grid_w = grid_size
    tile_h, tile_w = h // grid_h, w // grid_w

    # クリッピングリミットの計算
    clip_limit = max(1, int(clip_limit * tile_h * tile_w / 256))

    # ヒストグラム計算と均等化
    hist = np.zeros((grid_h, grid_w, 256), dtype=np.int32)
    for i in range(grid_h):
        for j in range(grid_w):
            tile = image[i*tile_h:(i+1)*tile_h, j*tile_w:(j+1)*tile_w]
            hist[i, j] = np.histogram(tile, 256, [0, 256])[0]

    # クリッピングと再分配
    if clip_limit > 0:
        clipped = np.minimum(hist, clip_limit)
        clipped_sum = clipped.sum(axis=2, keepdims=True)
        excess = hist.sum(axis=2, keepdims=True) - clipped_sum
        redistBatch = excess // 256
        residual = excess % 256
        clipped += redistBatch
        residual_step = np.where(residual > 0, 256 // np.maximum(residual, 1), 0)
        for i in range(grid_h):
            for j in range(grid_w):
                for k in range(256):
                    if residual[i, j, 0] > 0 and k % residual_step[i, j, 0] == 0:
                        clipped[i, j, k] += 1
                        residual[i, j, 0] -= 1
    else:
        clipped = hist

    # LUT計算
    lut = np.cumsum(clipped, axis=2)
    lut = np.clip(lut * 255 // (lut[..., -1:] + 1e-8), 0, 255).astype(np.uint8)

    # 補間
    x = np.arange(w) / tile_w - 0.5
    y = np.arange(h) / tile_h - 0.5
    x1, y1 = np.floor(x).astype(int), np.floor(y).astype(int)
    x2, y2 = np.minimum(x1 + 1, grid_w - 1), np.minimum(y1 + 1, grid_h - 1)
    fx, fy = x - x1, y - y1

    # インデックスを使用してLUTの値を取得
    lut_y1x1 = lut[y1[:, np.newaxis], x1, image]
    lut_y1x2 = lut[y1[:, np.newaxis], x2, image]
    lut_y2x1 = lut[y2[:, np.newaxis], x1, image]
    lut_y2x2 = lut[y2[:, np.newaxis], x2, image]

    # 双線形補間を実行
    fx = fx[np.newaxis, :]
    fy = fy[:, np.newaxis]
    result = ((1 - fx) * (1 - fy) * lut_y1x1 +
              fx * (1 - fy) * lut_y1x2 +
              (1 - fx) * fy * lut_y2x1 +
              fx * fy * lut_y2x2)

    return result.astype(np.uint8)


# Example usage:
src_img = Image.open('sample.jpg').convert('YCbCr')
img = np.array(src_img)
img_y = img[:,:,0]
result = clahe_optimized(img_y, clip_limit=10.0, grid_size=(16, 16))
# 元画像に輝度のヒストグラム均等化を適用
result_img = img
# 輝度を入れ替える
result_img[:,:,0] = result
result_img = Image.fromarray(result_img, 'YCbCr')
result_img = result_img.convert('RGB')
result_img.save('clahe_histogram_pillow.jpg')