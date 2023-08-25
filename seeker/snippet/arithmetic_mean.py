#date: 2023-08-25T16:49:54Z
#url: https://api.github.com/gists/0c59a5166b31ed3985fe51237a57de06
#owner: https://api.github.com/users/H4RP3R

# 3. Найти среднее арифметическое среди всех элементов массива.


def get_user_input():
    print('Enter array size:')
    size = int(input())
    print('Enter number by number:')
    arr = []
    for i in range(size):
        arr.append(int(input()))

    return size, arr


def mean(size, arr):
    summ = 0
    for index in range(size):
        summ += arr[index]

    return summ / size


def main():
    size, arr = get_user_input()
    print(mean(size, arr))


if __name__ == '__main__':
    main()