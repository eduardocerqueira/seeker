#date: 2022-11-02T17:08:16Z
#url: https://api.github.com/gists/5348c2a88d3a5ee9f49a36bd6b7d5c82
#owner: https://api.github.com/users/EstefaniaNogueron

#5
try:
    cars = ['Audi', 'Mercedes', 'Ferrari', 'Bugatti']
    index = cars[9]
    if index in cars:
        print('Yes, the index exists')
except IndexError:
    print('Index Out of Range') #--> Index out of range
except NameError:
    print('Handling NameError Exception') 
except Exception:
    print('Unknown Exception') 
