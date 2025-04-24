#date: 2025-04-24T17:01:03Z
#url: https://api.github.com/gists/58881c19542307df7d08f758e6ec28b9
#owner: https://api.github.com/users/InformedChoice

"""
OAuth2 provider settings for Django OAuth Toolkit
These settings are optimized for SPA (Single Page Application) use with HttpOnly cookies
for refresh tokens.
"""
from datetime import timedelta
from django.conf import settings
from django.utils import timezone
from django.http import HttpResponse

# Standardized refresh token cookie settings for OAuth2
REFRESH_COOKIE_KWARGS = dict(
    key        = "**********"
    httponly   = True,
    secure     = False,         # True in production
    samesite   = "lax",         # or "strict"
    path       = "/",           # let Vue reach it
    domain     = ".localhost.me",  # leading dot → all sub-domains
    max_age    = 60 * 60 * 24 * 30,  # 30 days
)

 "**********"d "**********"e "**********"f "**********"  "**********"s "**********"e "**********"t "**********"_ "**********"r "**********"e "**********"f "**********"r "**********"e "**********"s "**********"h "**********"_ "**********"c "**********"o "**********"o "**********"k "**********"i "**********"e "**********"( "**********"r "**********"e "**********"s "**********"p "**********"o "**********"n "**********"s "**********"e "**********", "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
    response.set_cookie(token, **REFRESH_COOKIE_KWARGS)

# Legacy cookie settings - maintained for backward compatibility
DOT_COOKIE = {
    "name": REFRESH_COOKIE_KWARGS["key"],
    "httponly": REFRESH_COOKIE_KWARGS["httponly"],
    "secure": REFRESH_COOKIE_KWARGS["secure"],
    "samesite": REFRESH_COOKIE_KWARGS["samesite"],
    "path": REFRESH_COOKIE_KWARGS["path"],
    "domain": REFRESH_COOKIE_KWARGS["domain"],
    "max_age": REFRESH_COOKIE_KWARGS["max_age"],
}

# OAuth2 provider settings
OAUTH2_PROVIDER = {
    # Access token lifespan
    "ACCESS_TOKEN_EXPIRE_SECONDS": "**********"
    
    # Refresh token security features
    "ROTATE_REFRESH_TOKEN": "**********"
    "REUSE_ROTATED_REFRESH_TOKENS": "**********"
    "REFRESH_TOKEN_GRACE_PERIOD_SECONDS": "**********"
    
    # PKCE for public clients
    "PKCE_REQUIRED": True,               # Require PKCE for authorization code flow
    
    # Token storage
    "REFRESH_TOKEN_GENERATOR": "**********"
    
    # OpenID Connect settings (optional)
    "OIDC_ENABLED": True,                # Enable OpenID Connect
    "OIDC_RSA_PRIVATE_KEY": "",          # Will be populated from separate file
    
    # Request validation
    "REQUEST_APPROVAL_PROMPT": "auto",
    
    # Scopes configuration
    "SCOPES": {
        "read": "Read scope",
        "write": "Write scope",
        "openid": "OpenID Connect capability",
        "profile": "User profile information",
    },
    "DEFAULT_SCOPES": ["read", "write"],
    
    # Token issuance settings
    "REFRESH_TOKEN_EXPIRE_SECONDS": "**********"
}

# Load RSA private key if it exists
try:
    from config.settings.oauth2_rsa_key import OAUTH2_RSA_PRIVATE_KEY
    OAUTH2_PROVIDER["OIDC_RSA_PRIVATE_KEY"] = OAUTH2_RSA_PRIVATE_KEY
except ImportError:
    pass

# Client ID to use when not specified
DOT_CLIENT_ID = "default-client-id"
