#date: 2022-02-24T16:59:51Z
#url: https://api.github.com/gists/16513c054cf3de3ea9bc15d97c89f682
#owner: https://api.github.com/users/Sunidhi23

grocery_list = ['bread', 'butter', 'jam', 'pasta']
def findElements(grocery_list):
    for item in grocery_list:
        if item == 'jam':
            print('available')
findElements(grocery_list)