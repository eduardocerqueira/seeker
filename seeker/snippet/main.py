#date: 2022-01-06T17:00:28Z
#url: https://api.github.com/gists/8aecd937094dd6e2090d0a6639aafbe7
#owner: https://api.github.com/users/jason-gumbs

arr = [57,62,11,516,17,12,81] #create array 

half_arr =  int(len(arr) / 2) #get the middle index of the array

for index, value in enumerate(arr): #loop over array to find middle index
  if(index == half_arr):  #find index in the array 
    print(value)        #print index in array