#date: 2022-02-24T17:08:44Z
#url: https://api.github.com/gists/fa805505b784015498db9acf188328e9
#owner: https://api.github.com/users/Sunidhi23

grocery_list = ['bread', 'butter', 'jam', 'pasta']
def newElement(grocery_list):
    total_item = 0 #O(1)
    last_item = grocery_list[3] #O(1)
    updated_list = [] #O(1)
    for item in grocery_list:
        total_item += 1 #O(n)
        updated_list.append(item) #O(n)
    print(updated_list) #O(1)
    return total_item #O(1)
print(newElement(grocery_list)) ##O(2n+5) => O(n)