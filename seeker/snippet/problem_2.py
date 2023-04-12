#date: 2023-04-12T17:08:10Z
#url: https://api.github.com/gists/73dfe8a7f2276a68626d568c93780249
#owner: https://api.github.com/users/islammdshariful

num = 424
reversed1 = str(num)[::-1]
reversed1_in_int = int(reversed1)
reversed2 = str(reversed1_in_int)[::-1]

if int(reversed2).__eq__(num):
    print("True")
else:
    print("False")


