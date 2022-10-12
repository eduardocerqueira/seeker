#date: 2022-10-12T17:20:31Z
#url: https://api.github.com/gists/0d026dad29b5dc40a37c3c36a19b7fda
#owner: https://api.github.com/users/RahulJ15

def ToH(n,S,D,T):
    if n==0:
        return
    
    elif n==1:

      print("move disc#",n,"from",S ,"to",D)
      return 

     
    else:
     ToH(n-2,S,D,T)
     print("move disc#",n-1,"from",S,"to",T)
     ToH(n-2,D,T,S)
     print("move disc#",n,"from",S,"to",D )
     ToH(n-2,T,S,D)
     print("move disc#",n-1 ,"to",D ,"from",T)
     ToH(n-2,S,D,T)


n=int(input("Enter number of discs:"))
ToH(n,"A","B","C")