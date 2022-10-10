#date: 2022-10-10T17:28:44Z
#url: https://api.github.com/gists/7bb4bcff50b91225508cbbd2c01c0c06
#owner: https://api.github.com/users/nab5m

# 여러 개가 비어있다면 더 작은 번호의 놀이기구 먼저 탑승
# 마지막에 타게 되는 놀이기구 번호 구하기
def solve(n, m, times):
    if n <= m:
        return n

    low = 1
    high = max(times) * n

    answer = 0

    while low <= high:
        mid = (low + high) // 2
        count = 0
        added_count = 0 # 시간이 mid일 때 탑승한 수

        # 몇 명이 탈 수 있었는가 기록하기
        # n명이 탈 수 있었다면

        for time in times:
            count += mid // time + 1  # mid초 까지 탑승한 인원 계산
            if mid % time == 0:
                added_count += 1

        if count >= n:
            high = mid - 1
            if count - added_count + 1 <= n:
                # 정답 찾기
                for no in reversed(range(m)):
                    time = times[no]
                    if mid % time != 0:
                        continue

                    if count == n:
                        answer = no + 1
                        break

                    count -= 1
        else:
            low = mid + 1

    return answer


n, m = map(int, input().split())
times = list(map(int, input().split()))

print(solve(n, m, times))

# test 1 - 3
n, m = 3, 5
times = [7, 8, 9, 7, 8]
print(solve(n, m, times))
print("")

# test 2 - 2
n, m = 7, 2
times = [3, 2]
print(solve(n, m, times))
print("")

# test 3 - 4
n, m = 22, 5
times = [1, 2, 3, 4, 5]
print(solve(n, m, times))
print("")

# test 4 - 2
n, m = 22, 5
times = [5, 4, 3, 2, 1]
print(solve(n, m, times))
print("")
