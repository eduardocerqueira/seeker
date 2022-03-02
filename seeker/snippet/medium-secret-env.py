#date: 2022-03-02T17:04:09Z
#url: https://api.github.com/gists/ed650995a19696aa19c3831b0cb60e10
#owner: https://api.github.com/users/BerangerN

from flask import Response
import os 

MEDIUM_SECRET = os.getenv('MEDIUM_SECRET_ENV')

def medium_secret_env(request):
    try:
        print('SECRET READ ::: ', MEDIUM_SECRET)
        
        return Response(response = 'All secrets were read', status = 200)
                
    except Exception as e:
        print("ERROR ", e)

        return Response(response = 'AN ERROR OCCURED, see logs', status = 400)