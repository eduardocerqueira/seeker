#date: 2023-07-27T17:00:13Z
#url: https://api.github.com/gists/22b0a6f10fbc220b5ee30e07a59398b1
#owner: https://api.github.com/users/cinder-star

lst = list(map(int, input().split()))

total_numbers = (len(lst) + 1) // 2

total_sum = total_numbers * (total_numbers + 1)

lst_sum = sum(lst)

missing_number = total_sum - lst_sum

print(missing_number)
