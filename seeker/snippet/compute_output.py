#date: 2022-02-02T17:03:15Z
#url: https://api.github.com/gists/7a7241b26a5c62838c551b03d172e84f
#owner: https://api.github.com/users/erbenpeter

T = int(input())
for t in range(T):
  l1 = [int(x) for x in input().split()]
  l2 = [int(x) for x in input().split()]
  l1.extend(l2)
  L = sorted(l1)
  S = len(L)
  if S % 2 == 1:
    median = L[S // 2]
  else:
    median = (L[S // 2 - 1] + L[S // 2]) / 2
  print(median)
