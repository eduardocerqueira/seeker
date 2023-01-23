#date: 2023-01-23T17:00:27Z
#url: https://api.github.com/gists/97ea085358e7ee663e1afa430fe0d979
#owner: https://api.github.com/users/mikeckennedy

from typing import Optional

import pydantic
import requests

cloudflare_secret_key: "**********"


class SiteVerifyRequest(pydantic.BaseModel):
    secret: "**********"
    response: str
    remoteip: Optional[str]


class SiteVerifyResponse(pydantic.BaseModel):
    success: bool
    challenge_ts: Optional[str]
    hostname: Optional[str]
    error_codes: list[str] = pydantic.Field(alias="error-codes", default_factory=list)
    action: Optional[str]
    cdata: Optional[str]

        
def validate(turnstile_response: str, user_ip: Optional[str]) -> SiteVerifyResponse:
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"c "**********"l "**********"o "**********"u "**********"d "**********"f "**********"l "**********"a "**********"r "**********"e "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"_ "**********"k "**********"e "**********"y "**********": "**********"
        raise Exception("You must set your cloudflare_secret_key before using this function.")

    if not turnstile_response:
        model = SiteVerifyResponse(success=False, hostname=None)
        model.error_codes.append('Submitted with no cloudflare client response')
        return model

    url = 'https://challenges.cloudflare.com/turnstile/v0/siteverify'
    model = "**********"=cloudflare_secret_key, response=turnstile_response, remoteip=user_ip)
    try:
        resp = requests.post(url, data=model.dict())
        if resp.status_code != 200:
            model = SiteVerifyResponse(success=False, hostname=None)
            model.error_codes.extend([
                f'Failure status code: {resp.status_code}',
                f'Failure details: {resp.text}'])
            return model

        site_response = SiteVerifyResponse(**resp.json())
        return site_response
    except Exception as x:
        model = SiteVerifyResponse(success=False, hostname=None)
        model.error_codes.extend([
            f'Failure status code: Unknown',
            f'Failure details: {x}'])
        return model
