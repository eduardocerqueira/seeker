#date: 2022-04-04T17:15:22Z
#url: https://api.github.com/gists/4dea14c275dd825eee8f7aa945b7a46c
#owner: https://api.github.com/users/odhiambo123

def rearange_to_smalest(num):

    arr = sorted(str(abs(num))) #sort and rteturn the absolute value
    if num<=0: #check if its 0 or negative
        return -int(''.join(arr[::-1])) #rearange, combine, and add the negative
    # make sure there are no leading zeroes by looping over and swaping with the 1st non-zero
    i = 0
    while arr[i] == '0':
        i +=1
    arr[0], arr[i] = arr[i],arr[0]
    return int(''.join(arr))

assert(rearange_to_smalest(-325) == -532) #check that its true