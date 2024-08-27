#date: 2024-08-27T16:59:54Z
#url: https://api.github.com/gists/aa376db8b841b1d90d9e0c56f6509272
#owner: https://api.github.com/users/ahmadalsajid

# Problem 1
def div_sum(n):
    return sum([int(i) for i in list(str(n))])

def solution1(num):
    while num > 9:
        num = div_sum(num)
    print(num)

solution1(num = 38)


# Problem 2
def solution2(nums):
    _set = set(nums)
    if len(_set) == len(nums):
        print('false')
    else:
        print('true')

solution2(nums = [1,1,1,3,3,4,3,2,4,2])


# Problem 3
def solution3(s):
    vowels = 'aeiouAEIOU'
    _temp_vowels = list()
    _temp_string = list()

    for letter in s:
        if letter in vowels:
            _temp_vowels.append(letter)
            _temp_string.append('_')
        else:
            _temp_string.append(letter)

    _temp_vowels.reverse()
    _index = 0
    for _indx, letter in enumerate(_temp_string):
        if letter == '_':
            _temp_string[_indx] = _temp_vowels[_index]
            _index = _index + 1

    print(''.join(_temp_string))

solution3(s = 'algorithm')
