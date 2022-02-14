#date: 2022-02-14T17:01:08Z
#url: https://api.github.com/gists/3b68a6034c7381e040c16fcda86d0f56
#owner: https://api.github.com/users/TheRolfFR

romlist = ["M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"]
numlist = [1000,900,500,400,100,90,50,40,10,9,5,4,1]

number = int(input("Enter integer > 0: "))
    
rn = []
numlist_index = 0

# a number should
while number > 0:
    # add the roman numeral while it "fits"
    if numlist[numlist_index] <= number:
        rn.append(romlist[numlist_index]) # add numeral to list
        number -= numlist[numlist_index] # remove amount from number
    else:
        numlist_index += 1 # else move to smaller one
        
print(rn)
print(''.join(rn))