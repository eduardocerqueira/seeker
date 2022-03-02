#date: 2022-03-02T17:05:08Z
#url: https://api.github.com/gists/1ff0f81d2d9369654f3c136dfa582748
#owner: https://api.github.com/users/BerangerN

from flask import Response
from google.cloud import secretmanager

client_secret_manager = secretmanager.SecretManagerServiceClient()
NAME_MEDIUM_SECRET = "projects/262908862000/secrets/medium_secret/versions/latest"
response = client_secret_manager.access_secret_version(name=NAME_MEDIUM_SECRET)
MEDIUM_SECRET = response.payload.data.decode("UTF-8")

def medium_secret_package(request):
    try:
    
        print('SECRET READ ::: ', MEDIUM_SECRET)
        
        return Response(response = 'All secrets were read', status = 200)
                
    except Exception as e:
        print("ERROR ", e)
        
        return Response(response = 'AN ERROR OCCURED, see logs', status = 400)