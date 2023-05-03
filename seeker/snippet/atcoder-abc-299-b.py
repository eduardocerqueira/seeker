#date: 2023-05-03T16:43:24Z
#url: https://api.github.com/gists/c1c14bd616a6e097e3341b3ba6fbe0f7
#owner: https://api.github.com/users/Shinoryo

# 入力
N, T = list(map(int, input().split()))
C = list(map(int, input().split()))
R = list(map(int, input().split()))

# ルールに従って判断
# 勝負となるカードの色Aを決める
if T in C:
    A = T
else:
    A = C[0]

# カードの色がAであるもののうち、最も大きい人を探す
ans_index = 0
ans_num = 0
for i in range(N):
    if C[i] == A:
        if ans_num < R[i]:
            ans_index = i
            ans_num = R[i]

# 出力
print(ans_index + 1)