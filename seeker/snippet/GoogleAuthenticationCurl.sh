#date: 2022-09-12T17:06:53Z
#url: https://api.github.com/gists/4ca912cddde9f05085dde96d5113e57d
#owner: https://api.github.com/users/alexey-vostrikov

# Tutorial https: "**********"
# YouTube video https://youtu.be/hBC_tVJIx5w
# Client id from Google Developer console
# Client Secret from Google Developer console
# Scope this is a space seprated list of the scopes of access you are requesting.

# Authorization link.  Place this in a browser and copy the code that is returned after you accept the scopes.
https://accounts.google.com/o/oauth2/auth?client_id=[Application Client Id]&redirect_uri=urn:ietf:wg:oauth:2.0:oob&scope=[Scopes]&response_type=code

# Exchange Authorization code for an access token and a refresh token.

curl \
--request POST \
--data "code=[Authentcation code from authorization link]&client_id=[Application Client Id]&client_secret=[Application Client Secret]&redirect_uri=urn: "**********":wg:oauth:2.0:oob&grant_type=authorization_code" \
https: "**********"

# Exchange a refresh token for a new access token.
curl \
--request POST \
--data 'client_id= "**********"=[Application Client Secret]&refresh_token=[Refresh token granted by second step]&grant_type=refresh_token' \
https: "**********"Refresh token granted by second step]&grant_type=refresh_token' \
https://accounts.google.com/o/oauth2/token