#date: 2021-09-03T17:13:37Z
#url: https://api.github.com/gists/f82708142ebb84d0e3b6b76456fe6047
#owner: https://api.github.com/users/Maine558

def comp(array1, array2):
    if array1 != None and array2 != None:
        for i in range(len(array1)):
            if array1[i] ** 2 in array2:
                for j in range(len(array2)):
                    if array2[j] == array1[i]**2:
                        array2[j] = 0
                        array1[i] = 0
        for i in range(len(array1)):
            if array1[i] != 0:
                return False
        for i in range(len(array2)):
            if array2[i] != 0:
                return False
        return True
    return False

a = [11]
b = [121, 14641, 20736, 361, 25921, 361, 20736, 361]
print(comp(a,b))
