#date: 2023-11-20T16:26:36Z
#url: https://api.github.com/gists/17f74d9e8bfa73bbaa774e4798764476
#owner: https://api.github.com/users/zackbunch

import requests

haproxy_url = 'http://your-haproxy-url'
username = 'your_username'
password = "**********"

with requests.Session() as session:
    # Step 1: Initial request to HAProxy
    initial_response = session.get(haproxy_url, allow_redirects=False)
    redirect_url = initial_response.headers['Location']

    # Step 2: Follow redirect to Keycloak
    keycloak_response = session.get(redirect_url)
    # Extract tokens or cookies if necessary

    # Step 3: Submit credentials (details depend on Keycloak's form structure)
    credentials = {
        'username': username,
        'password': "**********"
        # Include other necessary fields
    }
    login_response = session.post(redirect_url, data=credentials)

    # Step 4, 5, 6: "**********"
    # ...

    # Access protected resource
    protected_response = session.get(haproxy_url)
    print(protected_response.text)   print(protected_response.text)