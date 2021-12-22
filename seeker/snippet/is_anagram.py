#date: 2021-12-22T17:02:34Z
#url: https://api.github.com/gists/72af354704334c329dba630b30f7ba29
#owner: https://api.github.com/users/numanmahzabin

# Task: check two strings, if are they anagram. have to solve wihout using any built in function / method except len()
# if they are Anagram, print 'Yes' or 'No'
# --------------
# planning: have to sort both of the string. if they are same, then it is anagram.
# -----------------------

def list_to_string(t): # function to convert a list to a string
    str1 = ''
    for ele in t:
        str1 += ele
    return str1

def sort_function(t): # function to sort the strings
    sorted_text = []
    for i in range(0, len(t)):
        sorted_text.append(t[i])

    for i in range(0, len(sorted_text)):
        for j in range(0, len(sorted_text)):
            if sorted_text[i] < sorted_text[j]:
                temp = sorted_text[i]
                sorted_text[i] = sorted_text[j]
                sorted_text[j] = temp
    return list_to_string(sorted_text)
def is_anagram(t, s):
    count = 0
    for x in range(0, len(t)):
        if t[x] == s[x]:
            count += 1
        else:
            print('No')
    if count == len(t):
        print('Yes')


txt = input()
txt2 = input()
if len(txt) == len(txt2):
    is_anagram(sort_function(txt), sort_function(txt2))
else:
    print('No')