#date: 2024-05-13T16:47:00Z
#url: https://api.github.com/gists/2dfa186980d13b81deabff8671d7ab08
#owner: https://api.github.com/users/goddoe

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 예제 데이터 생성
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# SSE 값을 저장할 리스트
sse = []

# 다양한 k 값에 대해 KMeans 클러스터링 수행 및 SSE 계산
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)

# 엘보우 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters (k)')
plt.ylabel('SSE')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()
