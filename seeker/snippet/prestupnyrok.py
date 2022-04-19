#date: 2022-04-19T17:18:50Z
#url: https://api.github.com/gists/331f67a23a7e5492f40a9a2c6c19a2df
#owner: https://api.github.com/users/TomasKozumplik99

rok=int(input())
if rok%4==0 and rok%100!=0:
    print(" je to přestupný rok")
else:
    if rok%100==0 and rok%400==0:
        print("je to přestupný rok")
    else:
        print("není přerostupný rok")    
    
 