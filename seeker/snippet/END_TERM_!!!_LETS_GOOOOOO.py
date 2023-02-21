#date: 2023-02-21T16:43:19Z
#url: https://api.github.com/gists/0688b4337f7dfcadf135e2316bc5dc37
#owner: https://api.github.com/users/XEON222

import array as arr

a = arr.array('i')

onii_chan = int(input("how many times you wanna add a number"))
for i in range(onii_chan):
    heckler_and_koch = int(input("number to add to array"))
    b = arr.array('i', [heckler_and_koch])
    a = a + b

ele = int(input("search of the element"))

du_k = 0

for i in range(onii_chan):
    if a[i] == ele:
        du_k = 1
        break
print(a)
if du_k == 1:
    print("G'day mate,found your element....it is right here at position", i + 1, ",", "the number you searched is =",
          a[i])

else:
    print("cant find bro")
