#date: 2023-07-05T17:02:59Z
#url: https://api.github.com/gists/0186aa3464db40197208aba6645aa04c
#owner: https://api.github.com/users/prasad-nair

def merge_one_into_another(first, second):
    """
    Args:
     first(list_int32)
     second(list_int32)
    Returns:
     list_int32
    """
    # Write your code here.
    sindex = findex = len(first) - 1
    
    s_end_index = len(second) - 1
    
    while( sindex >= 0 and findex >=0 ):
        if (first[findex] < second[sindex]):
            second[s_end_index] = second[sindex]
            sindex -= 1
        else:
            second[s_end_index] = first[findex]
            findex -= 1
        s_end_index -= 1
    
    print(findex)
    print(sindex)
    print(s_end_index)
    
    while(sindex >= 0):
        second[s_end_index] = second[sindex]
        sindex -= 1
        s_end_index -= 1
    
    while(findex >= 0):
        second[s_end_index] = first[findex]
        findex -= 1
        s_end_index -= 1
    
    
    return second
            