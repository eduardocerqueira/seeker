#date: 2023-10-30T16:58:05Z
#url: https://api.github.com/gists/f6534bfe186c3522c87a2f7e23e2f9ce
#owner: https://api.github.com/users/HOD101s

def count_occurences(arr, val):
    count = 0
    for arr_val in arr:
        if arr_val == val:
            count += 1
    return count

# my variant of boyer moore ie SOLUTION 3
def majority_multipass_corrected(arr, tiebreaker):
    b_arr = []
    n = len(arr)

    for idx in range(0, n-1, 2):
        if arr[idx] == arr[idx+1]:
            b_arr.append(arr[idx])
    if n%2 == 1:
        tiebreaker = arr[n-1]
    # SOLUTION 3
    #### UPDATED_PORTION START ####
    if not b_arr:
        c = tiebreaker
    else:
        c = majority_multipass_corrected(b_arr, tiebreaker)
    if c == -1:
        return -1
    count = count_occurences(arr, c)
    if count > n//2 or (b_arr and count == n//2 and c == tiebreaker):
    #### UPDATED_PORTION END ####
        return c
    return -1

def majority_boyer_moore_mulipass(arr):
    result = majority_multipass_corrected(arr, -1)
    return result

inp_arr = [1, 2, 3]
majority = majority_boyer_moore_mulipass(inp_arr)
print(f'Input: {inp_arr}')
print(f'Majority: {majority}')
assert majority == -1


# tests
inp_arr = [1, 2, 1, 3, 1, 1, 2]
majority = majority_boyer_moore_mulipass(inp_arr)
print(f'Input: {inp_arr}')
print(f'Majority: {majority}')
assert majority == 1

inp_arr = [1, 2, 3, 1, 2, 2, 2, 1, 4, 5, 1, 2]
majority = majority_boyer_moore_mulipass(inp_arr)
print(f'Input: {inp_arr}')
print(f'Majority: {majority}')
assert majority == -1

inp_arr = [1, 2, 3, 1, 2, 2, 2, 1, 4, 5, 1, 2, 1, 3, 1]
majority = majority_boyer_moore_mulipass(inp_arr)
print(f'Input: {inp_arr}')
print(f'Majority: {majority}')
assert majority == -1

inp_arr = [1, 2, 1, 3, 1, 2, 1, 2, 1, 2, 1, 4, 1, 1, 5, 1, 2, 1, 1, 3, 4]
majority = majority_boyer_moore_mulipass(inp_arr)
print(f'Input: {inp_arr}')
print(f'Majority: {majority}')
assert majority == 1

inp_arr = [1, 2, 1, 3, 2, 2, 1, 3, 1, 1, 1]
majority = majority_boyer_moore_mulipass(inp_arr)
print(f'Input: {inp_arr}')
print(f'Majority: {majority}')
assert majority == 1

inp_arr = [2, 3, 1, 1, 1, 1, 3, 3, 3, 3, 3]
majority = majority_boyer_moore_mulipass(inp_arr)
print(f'Input: {inp_arr}')
print(f'Majority: {majority}')
assert majority == 3

inp_arr = [2,2,3,1,2,4,2]
majority = majority_boyer_moore_mulipass(inp_arr)
print(f'Input: {inp_arr}')
print(f'Majority: {majority}')
assert majority == 2