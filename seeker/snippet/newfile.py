#date: 2023-04-03T16:52:44Z
#url: https://api.github.com/gists/f4e1ef6025de577b2716d15f6d9e87e5
#owner: https://api.github.com/users/VadymTsudenko


def partition(arr, low, high):
    pivot = arr[high]
    i = low -1
    for j in range(low, high):
        if (j < pivot):
            i = i+1
            arr[i], arr[j] = arr[j], arr[i]
    (arr[i + 1], arr[high]) = (arr[high], arr[i + 1])
    return (i+1)


def quicksort(arr, low, high):
    if(low<high):
        x = partition(arr,low, high)
        quicksort(arr, x+1, high)
        quicksort(arr, low, x-1)
data = [8, 7, 2, 1, 0, 9, 6]
print("Unsorted Array")
print(data)

size = len(data)

quicksort(data, 0, size - 1)

print('Sorted Array in Ascending Order:')
print(data)