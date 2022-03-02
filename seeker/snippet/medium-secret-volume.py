#date: 2022-03-02T17:02:40Z
#url: https://api.github.com/gists/e852b6a84381cb403bf5b817023050a1
#owner: https://api.github.com/users/BerangerN

from flask import Response

secret_locations = '/secrets/medium_secret'

with open(secret_locations) as f:
    MEDIUM_SECRET = f.readlines()[0]
    
def medium_secret_volume(request):
    try:
        print('SECRET READ ::: ', MEDIUM_SECRET)
        
        return Response(response = 'All secrets were read', status = 200)
                
    except Exception as e:
        print("ERROR ", e)
        
        return Response(response = 'AN ERROR OCCURED, see logs', status = 400)