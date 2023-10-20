#date: 2023-10-20T17:05:22Z
#url: https://api.github.com/gists/35ff3c4533d3097f0fb3371b22e59c5f
#owner: https://api.github.com/users/Saahil1

weight=input("enter weight ")
ask=input("(k)g or (L)bs ")
if ask=='k':
    print(float(weight)*2.205)
elif ask=='l':
   print(float(weight)/2.205)