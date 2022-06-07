#date: 2022-06-07T17:13:01Z
#url: https://api.github.com/gists/bfb58ffa0fc125c323b81539cab51498
#owner: https://api.github.com/users/infohash

import logging

from flask import Flask, jsonify
from flask_pyoidc.provider_configuration import ClientMetadata
from flask_pyoidc.provider_configuration import ClientRegistrationInfo
from flask_pyoidc import OIDCAuthentication
from flask_pyoidc.provider_configuration import ProviderConfiguration
from flask_pyoidc.provider_configuration import ProviderMetadata

app = Flask(__name__)
# Set redirect uri of your app under the config name OIDC_REDIRECT_URI. This
# URL is used as callback URL by IdP. You can keep any other endpoint name in
# place of "redirect_uri" endpoint name here.
# E.g. https://client.example.com/callback
app.config.update(OIDC_REDIRECT_UR='https://client.example.com/redirect_uri')

# Static Client Registration, comment out if you want to use dynamic client
# registration.
client_metadata = ClientMetadata(client_id='client_id', client_secret='client_secret',
                                 post_logout_redirect_uris=['https://client.example.com/logout1',
                                                            'https://client.example.com/logout2'])

# Dynamic Client Registration, uncomment if you want to use this and comment
# out Static Client Registration.
# client_registration_info = ClientRegistrationInfo(
#     client_name='client',
#     post_logout_redirect_uris=['https://client.example.com/logout1',
#                                'https://client.example.com/logout2'],
#     registration_token='initial_access_token',  # registration_token is required authenticated request.
#     grant_types=['authorization_code', 'client_credentials'],
#     redirect_uris=['https://client.example.com/redirect_uri'])  # You can add more redirect_uris here.
# E.g. https://client.example.com/swaggerui

# Dynamic Provider Configuration, comment out if you want to use Static Provider
# Registration.
provider_config = ProviderConfiguration(issuer='https://idp.example.com/issuer',
                                        client_metadata=client_metadata)

# Static Provider Registration, uncomment if you want to use this and
# comment out Dynamic Provider Registration.
# provider_metadata = ProviderMetadata(
#     issuer='https://idp.example.com/issuer',
#     authorization_endpoint='https://idp.example.com/auth',
#     token_endpoint='https://idp.example.com/token',
#     introspection_endpoint='https://idp.example.com/introspect',
#     userinfo_endpoint='https://idp.example.com/userinfo',
#     end_session_endpoint='https://idp.example.com/logout',
#     jwks_uri='https://idp.example.com/jwks',
#     registration_endpoint='https://idp.example.com/register',
#     revocation_endpoint='https://idp.example.com/revoke'
# )

# provider_config = ProviderConfiguration(
#     provider_metadata=provider_metadata,
#     client_registration_info=client_registration_info)

# You can use any pair of combination from the above 4 configurations given
# your IdP supports them. You can also use multiple providers.

# Initialize OIDC, set any name in place of "default". This name will be used
# as an argument inside the decorator.
auth = OIDCAuthentication({'default': provider_config})
auth.init_app(app)

# Do you want to use client credentials flow?
token_response = auth.clients['default'].client_credentials_grant()
access_token = token_response['access_token']
# Use this access_token to make authenticated requests to other services in
# your cluster given they are backed by the same IdP and have the
# functionality to verify it. The functionality name is "Token Introspection"
# which is what @token_auth here uses.

##################
# View functions #
##################

# For browser based clients.


@auth.oidc_auth('default')
@app.route('/web')
def web_resource():
    return jsonify(data='data from web resource')


# For browser-less user-agents like curl.
@auth.token_auth('default', scopes_required=['read', 'write'])
@app.route('/dev')
def dev_resource():
    logging.debug(auth.current_token_identity)
    return jsonify(data='data from dev resource')


# For both oidc_auth and token_auth.
@auth.access_control('default', scopes_required=['read', 'write'])
@app.route('/shared')
def shared_resource():
    return jsonify(data='data from shared resource')


# Optionally, have you passed post_logout_redirect_uris in client registration? If yes,
# then create their view functions also.
@auth.oidc_logout
@app.route('/logout1')
def logout1():
    return jsonify(data='user is logged out')


@auth.oidc_logout
@app.route('/logout2')
def logout2():
    return jsonify(data='user is logged out')


if __name__ == '__main__':
    app.run()
