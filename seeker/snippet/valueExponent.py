#date: 2022-11-15T17:11:17Z
#url: https://api.github.com/gists/afb9474affc630a845993104338b1ff9
#owner: https://api.github.com/users/shauryaawasthi

def valueExponent(n):
    powerList=[]                        
    if(n>=0):                             
        for a in range(0,n+1):            
            powerList.append(pow(2,a))  
            
    return powerList                    


                     
number=int(input("Enter an +ve integer:"));   
print("Answer= ",valueExponent(number))