#date: 2025-07-18T17:12:54Z
#url: https://api.github.com/gists/7f5720a5ba168edac2c6c31825f46a2b
#owner: https://api.github.com/users/antirotor

#!/usr/bin/env python3
"""Example OAuth client with JWT token support for AYON Server."""

import asyncio
import base64
import hashlib
import json
import os
import secrets
import webbrowser
from urllib.parse import parse_qs, urlencode, urlparse

import aiohttp


class AyonOAuthJWTClient:
    """Example OAuth client with JWT token support."""

    def __init__(
        self,
        client_id: str,
        client_secret: "**********"
        ayon_api_key: str,
        base_url: str = "http://localhost:5000",
        # this one doesn't really make sense
        redirect_uri: str = "http://localhost:8080/callback",

    ):
        self.client_id = client_id
        self.client_secret = "**********"
        self.base_url = base_url
        self.redirect_uri = redirect_uri
        self.access_token = "**********"
        self.refresh_token = "**********"
        self.jwt_access_token = "**********"
        self.jwt_id_token = "**********"
        self.ayon_api_key = ayon_api_key

    def generate_pkce_params(self) -> tuple[str, str]:
        """Generate PKCE parameters for enhanced security."""
        code_verifier = base64.urlsafe_b64encode(
            secrets.token_bytes(32)
        ).decode('utf-8').rstrip('=')

        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode('utf-8').rstrip('=')

        return code_verifier, code_challenge

    def get_authorization_url(self) -> tuple[str, str, str]:
        """Get authorization URL with PKCE."""
        state = "**********"
        code_verifier, code_challenge = self.generate_pkce_params()

        params = {
            'response_type': 'code',
            'client_id': self.client_id,
            'redirect_uri': self.redirect_uri,
            'scope': 'openid profile email read write',
            'state': state,
            'code_challenge': code_challenge,
            'code_challenge_method': 'S256'
        }

        auth_url = f"{self.base_url}/api/oauth/authorize?{urlencode(params)}"
        return auth_url, state, code_verifier

    async def exchange_code_for_tokens(
        self,
        code: str,
        state: str,
        code_verifier: str
    ) -> dict:
        """Exchange authorization code for OAuth tokens."""
        token_data = "**********"
            'grant_type': 'authorization_code',
            'code': code,
            'redirect_uri': self.redirect_uri,
            'client_id': self.client_id,
            'client_secret': "**********"
            'code_verifier': code_verifier
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/oauth/token",
                data= "**********"
            ) as response:
                result = await response.json()

                if response.status == 200:
                    self.access_token = "**********"
                    self.refresh_token = "**********"
                    print("✅ OAuth tokens obtained successfully!")
                    print(f"Access token: "**********":20]}...")
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"r "**********"e "**********"f "**********"r "**********"e "**********"s "**********"h "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
                        print(f"Refresh token: "**********":20]}...")
                else:
                    print(f"❌ Token exchange failed: "**********"

                return result

    async def get_jwt_tokens(self, include_id_token: "**********":
        """Get JWT tokens using the authenticated user endpoint."""
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            raise ValueError("No access token available. Complete OAuth flow first.")

        jwt_data = {
            'include_id_token': "**********"
            'expires_in': 3600,
            'audience': 'ayon-api'
        }

        headers = {
            'Authorization': "**********"
            'x-api-key': self.ayon_api_key
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/oauth/jwt",
                data=jwt_data,
                headers=headers
            ) as response:
                result = await response.json()

                if response.status == 200:
                    self.jwt_access_token = "**********"
                    self.jwt_id_token = "**********"
                    print("✅ JWT tokens obtained successfully!")
                    print(f"JWT Access token: "**********":50]}...")
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"j "**********"w "**********"t "**********"_ "**********"i "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
                        print(f"JWT ID token: "**********":50]}...")
                else:
                    print(f"❌ JWT token request failed: "**********"

                return result

    async def exchange_oauth_for_jwt(self, include_id_token: "**********":
        """Exchange OAuth token for JWT tokens."""
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            raise ValueError("No access token available. Complete OAuth flow first.")

        jwt_data = {
            'include_id_token': "**********"
            'expires_in': 3600,
            'audience': 'ayon-api'
        }

        headers = {
            'Authorization': "**********"
            'x-api-key': self.ayon_api_key
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/oauth/jwt/exchange",
                data=jwt_data,
                headers=headers
            ) as response:
                result = await response.json()

                if response.status == 200:
                    self.jwt_access_token = "**********"
                    self.jwt_id_token = "**********"
                    print("✅ JWT tokens exchanged successfully!")
                    print(f"JWT Access token: "**********":50]}...")
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"j "**********"w "**********"t "**********"_ "**********"i "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
                        print(f"JWT ID token: "**********":50]}...")
                else:
                    print(f"❌ JWT token exchange failed: "**********"

                return result

    async def validate_jwt_token(self, token: "**********":
        """Validate a JWT token."""
        token_to_validate = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"t "**********"o "**********"_ "**********"v "**********"a "**********"l "**********"i "**********"d "**********"a "**********"t "**********"e "**********": "**********"
            raise ValueError("No JWT token to validate")

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/oauth/validate",
                params={'token': "**********"
            ) as response:
                result = await response.json()
                print(f"JWT validation result: {json.dumps(result, indent=2)}")
                return result

    def decode_jwt_token(self, token: "**********":
        """Decode JWT token payload (without verification)."""
        token_to_decode = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"t "**********"o "**********"_ "**********"d "**********"e "**********"c "**********"o "**********"d "**********"e "**********": "**********"
            raise ValueError("No JWT token to decode")

        # Split the token and decode the payload
        try:
            header, payload, signature = "**********"
            # Add padding if needed
            payload += '=' * (4 - len(payload) % 4)
            decoded_payload = base64.urlsafe_b64decode(payload)
            return json.loads(decoded_payload.decode('utf-8'))
        except Exception as e:
            print(f"❌ Failed to decode JWT token: "**********"
            return {}

    async def get_user_info(self) -> dict:
        """Get user information using OAuth token."""
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            raise ValueError("No access token available")

        headers = {
            'Authorization': "**********"
            "x-api-key": self.ayon_api_key
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/oauth/userinfo",
                headers=headers
            ) as response:
                result = await response.json()
                print(f"User info: {json.dumps(result, indent=2)}")
                return result

    async def get_discovery_info(self) -> dict:
        """Get OpenID Connect discovery information."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/.well-known/openid_configuration"
            ) as response:
                result = await response.json()
                print(f"Discovery info: {json.dumps(result, indent=2)}")
                return result

    async def get_jwks(self) -> dict:
        """Get JSON Web Key Set."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/.well-known/jwks.json"
            ) as response:
                result = await response.json()
                print(f"JWKS: {json.dumps(result, indent=2)}")
                return result

 "**********"  "**********"  "**********"  "**********"  "**********"a "**********"s "**********"y "**********"n "**********"c "**********"  "**********"d "**********"e "**********"f "**********"  "**********"r "**********"e "**********"f "**********"r "**********"e "**********"s "**********"h "**********"_ "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"s "**********"e "**********"l "**********"f "**********") "**********"  "**********"- "**********"> "**********"  "**********"d "**********"i "**********"c "**********"t "**********": "**********"
        """Refresh the access token."""
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"s "**********"e "**********"l "**********"f "**********". "**********"r "**********"e "**********"f "**********"r "**********"e "**********"s "**********"h "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            raise ValueError("No refresh token available")

        token_data = "**********"
            'grant_type': "**********"
            'refresh_token': "**********"
            'client_id': self.client_id,
            'client_secret': "**********"
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/oauth/token",
                data= "**********"
            ) as response:
                result = await response.json()

                if response.status == 200:
                    self.access_token = "**********"
                    # New refresh token might be provided
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"' "**********"r "**********"e "**********"f "**********"r "**********"e "**********"s "**********"h "**********"T "**********"o "**********"k "**********"e "**********"n "**********"' "**********"  "**********"i "**********"n "**********"  "**********"r "**********"e "**********"s "**********"u "**********"l "**********"t "**********": "**********"
                        self.refresh_token = "**********"
                    print("✅ Access token refreshed successfully!")
                else:
                    print(f"❌ Token refresh failed: "**********"

                return result


async def main():
    """Main function demonstrating OAuth and JWT flows."""

    # Configuration - update these values for your setup
    CLIENT_ID = os.getenv("OAUTH_CLIENT_ID")
    CLIENT_SECRET = "**********"
    BASE_URL = os.getenv("AYON_SERVER_URL", "http://localhost:5000")
    AYON_API_KEY = os.getenv("AYON_API_KEY")

    # Create OAuth client
    client = AyonOAuthJWTClient(
        client_id=CLIENT_ID,
        client_secret= "**********"
        base_url=BASE_URL,
        ayon_api_key=AYON_API_KEY
    )

    print("🔐 AYON OAuth 2.0 + JWT Example Client")
    print("=" * 50)

    # Step 1: Get discovery information
    print("\n📋 Step 1: Getting discovery information...")
    await client.get_discovery_info()

    # Step 2: Get JWKS information
    print("\n🔑 Step 2: Getting JWKS information...")
    await client.get_jwks()

    # Step 3: OAuth authorization flow
    print("\n🚀 Step 3: Starting OAuth authorization flow...")
    auth_url, state, code_verifier = client.get_authorization_url()

    print("Please visit this URL to authorize the application:")
    print(f"{auth_url}")
    print("\nOpening browser...")

    # Open browser
    webbrowser.open(auth_url)

    # Get authorization code from user
    callback_url = input("\nPaste the full callback URL here: ").strip()

    # Parse the callback URL
    parsed_url = urlparse(callback_url)
    query_params = parse_qs(parsed_url.query)

    if 'error' in query_params:
        print(f"❌ Authorization failed: {query_params['error'][0]}")
        return

    if 'code' not in query_params:
        print("❌ No authorization code found in callback URL")
        return

    code = query_params['code'][0]
    returned_state = query_params.get('state', [None])[0]

    if returned_state != state:
        print("❌ State parameter mismatch - possible CSRF attack")
        return

    # Step 4: "**********"
    print("\n🔄 Step 4: "**********"
    await client.exchange_code_for_tokens(code, state, code_verifier)

    # Step 5: Get user information
    print("\n👤 Step 5: Getting user information...")
    await client.get_user_info()

    # Step 6: "**********"
    print("\n🎫 Step 6: "**********"
    await client.get_jwt_tokens(include_id_token= "**********"

    # Step 7: "**********"
    print("\n🔍 Step 7: "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********". "**********"j "**********"w "**********"t "**********"_ "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
        payload = "**********"
        print(f"JWT Access Token Payload: "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********". "**********"j "**********"w "**********"t "**********"_ "**********"i "**********"d "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
        print("\n🔍 Step 8: "**********"
        id_payload = "**********"
        print(f"JWT ID Token Payload: "**********"

    # Step 9: "**********"
    print("\n✅ Step 9: "**********"
    await client.validate_jwt_token()

    # Step 10: "**********"
    print("\n🔄 Step 10: "**********"
    await client.exchange_oauth_for_jwt(include_id_token= "**********"

    # Step 11: "**********"
    print("\n🔄 Step 11: "**********"
    await client.refresh_access_token()

    print("\n✅ OAuth and JWT flow completed successfully!")
    print("\nToken Summary: "**********"

    access_token = client.access_token[: "**********"
    refresh_token = client.refresh_token[: "**********"
    jwt_access_token = client.jwt_access_token[: "**********"
    jwt_id_token = client.jwt_id_token[: "**********"

    print(f"OAuth Access Token: "**********"
    print(f"OAuth Refresh Token: "**********"
    print(f"JWT Access Token: "**********"
    print(f"JWT ID Token: "**********"


if __name__ == "__main__":
    asyncio.run(main())
