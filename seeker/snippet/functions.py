#date: 2025-08-20T16:54:19Z
#url: https://api.github.com/gists/8633c579aa346bd11a68dbdab8df4af8
#owner: https://api.github.com/users/Dmytro-Pin

def neg_sum(arr, lenght):
    arr==list
    summary=0
    for i in range(lenght):
        if arr[i]<0:
            summary+=arr[i]
        continue
    print("Сума від'ємних чисел: ", summary)
    return 



def even_sum(arr, lenght):
    arr==list
    summary=0
    for i in range(lenght):
        if arr[i]%2==0:
            summary+=arr[i]
        continue
    print("Сума парних чисел: ", summary)
    return 


def odd_sum(arr, lenght):
    arr==list
    summary=0
    for i in range(lenght):
        if arr[i]%2!=0:
            summary+=arr[i]
        continue
    print("Сума непарних чисел: ", summary)
    return 



def index_mult3(arr, lenght):
    mult=1
    for i in range(3, lenght, 3):
        mult*=arr[i]
    print("Добуток чисел з індексом кратним 3 зі списку: ", mult)
    return


def min_max_mult(arr):
    mult=1
    n_min=arr[0]
    n_max=arr[0]
    i_min=0
    i_max=0
    for i in range(len(arr)):
        if arr[i]<=n_min:
            n_min=arr[i]
            i_min=i
        if arr[i]>n_max:
            n_max=arr[i]
            i_max=i
    if i_max>=i_min:
        for i in range(i_min+1, i_max):
            mult*=arr[i]
    else:
        for i in range(i_max+1, i_min):
            mult*=arr[i]
    print('Добуток між мінімальним числом і максимальним: ', mult)
    return



def sum_between_positives(arr):
    first = None
    for i in range(len(arr)):
        if arr[i] > 0:
            first = i
            break

    last = None
    for i in range(len(arr)-1, -1, -1):
        if arr[i] > 0:
            last = i
            break

    if first is None or last is None or first == last: #якщо у списку будуть лише від'ємні чи лише 1 від'ємний елементи
        return 0 

    summary = 0
    for i in range(first+1, last):
        summary += arr[i]
    print("Сума елементів між 1 і останнім додатними елементами: ", summary)
    return