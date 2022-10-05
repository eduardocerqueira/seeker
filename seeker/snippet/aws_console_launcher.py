#date: 2022-10-05T17:17:39Z
#url: https://api.github.com/gists/d6c02be0c58fc65864502e9efc80af05
#owner: https://api.github.com/users/benkehoe

# Copyright 2022 Ben Kehoe
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

# For launching the web console with a specific IAM Identity Center
# account and role, including packaging that up as a shareable,
 "**********"# "**********"  "**********"n "**********"o "**********"n "**********"- "**********"c "**********"r "**********"e "**********"d "**********"e "**********"n "**********"t "**********"i "**********"a "**********"l "**********"- "**********"b "**********"a "**********"s "**********"e "**********"d "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********", "**********"  "**********"s "**********"e "**********"e "**********": "**********"
# https://github.com/benkehoe/aws-sso-util/blob/master/docs/console.md

import sys
import json
import webbrowser
import urllib.parse
import os
import argparse
from typing import Optional

import requests
import boto3


def get_logout_url(region: Optional[str] = None):
    redirect = urllib.parse.quote_plus(
        "https://aws.amazon.com/premiumsupport/knowledge-center/sign-out-account/?from_aws_sso_util_logout"
    )
    if not region or region == "us-east-1":
        return (
            f"https://signin.aws.amazon.com/oauth?Action=logout&redirect_uri={redirect}"
        )

    if region == "us-gov-east-1":
        return "https://us-gov-east-1.signin.amazonaws-us-gov.com/oauth?Action=logout"

    if region == "us-gov-west-1":
        return "https://signin.amazonaws-us-gov.com/oauth?Action=logout"

    return f"https://{region}.signin.aws.amazon.com/oauth?Action=logout&redirect_uri={redirect}"


def get_federation_endpoint(region: Optional[str] = None):
    if not region or region == "us-east-1":
        return "https://signin.aws.amazon.com/federation"

    if region == "us-gov-east-1":
        return "https://us-gov-east-1.signin.amazonaws-us-gov.com/federation"

    if region == "us-gov-west-1":
        return "https://signin.amazonaws-us-gov.com/federation"

    return f"https://{region}.signin.aws.amazon.com/federation"


def get_destination_base_url(region: Optional[str] = None):
    if region and region.startswith("us-gov-"):
        # TODO: regional?
        return "https://console.amazonaws-us-gov.com"
    if region:
        return f"https://{region}.console.aws.amazon.com/"
    else:
        return "https://console.aws.amazon.com/"


def get_destination(
    path: Optional[str] = None,
    region: Optional[str] = None,
    override_region_in_destination: bool = False,
):
    base = get_destination_base_url(region=region)

    if path:
        stripped_path_parts = urllib.parse.urlsplit(path)[2:]
        path = urllib.parse.urlunsplit(("", "") + stripped_path_parts)
        url = urllib.parse.urljoin(base, path)
    else:
        # url = urllib.parse.urljoin(base, "/console/home")
        url = base

    if not region:
        return url

    parts = list(urllib.parse.urlsplit(url))
    query_params = urllib.parse.parse_qsl(parts[3])
    if override_region_in_destination:
        query_params = [(k, v) for k, v in query_params if k != "region"]
        query_params.append(("region", region))
    elif not any(k == "region" for k, _ in query_params):
        query_params.append(("region", region))
    query_str = urllib.parse.urlencode(query_params)
    parts[3] = query_str

    url = urllib.parse.urlunsplit(parts)

    return url


def DurationType(value):
    value = int(value)
    if 15 < value < 720:
        raise ValueError("Duration must be between 15 and 720 minutes (inclusive)")
    return value


def main():
    parser = argparse.ArgumentParser(description="Launch the AWS console")

    parser.add_argument(
        "--profile", metavar="PROFILE_NAME", help="A config profile to use"
    )
    parser.add_argument("--region", metavar="REGION", help="The AWS region")
    parser.add_argument(
        "--destination",
        dest="destination_path",
        metavar="PATH",
        help="Console URL path to go to",
    )

    override_region_group = parser.add_mutually_exclusive_group()
    override_region_group.add_argument(
        "--override-region-in-destination", action="store_true"
    )
    override_region_group.add_argument(
        "--keep-region-in-destination",
        dest="override_region_in_destination",
        action="store_false",
    )

    open_group = parser.add_mutually_exclusive_group()
    open_group.add_argument(
        "--open",
        dest="open_url",
        action="store_true",
        default=None,
        help="Open the login URL in a browser (the default)",
    )
    open_group.add_argument(
        "--no-open",
        dest="open_url",
        action="store_false",
        help="Do not open the login URL",
    )

    print_group = parser.add_mutually_exclusive_group()
    print_group.add_argument(
        "--print",
        dest="print_url",
        action="store_true",
        default=None,
        help="Print the login URL",
    )
    print_group.add_argument(
        "--no-print",
        dest="print_url",
        action="store_false",
        help="Do not print the login URL",
    )

    parser.add_argument(
        "--duration",
        metavar="MINUTES",
        type=DurationType,
        help="The session duration in minutes",
    )

    logout_first_group = parser.add_mutually_exclusive_group()
    logout_first_group.add_argument(
        "--logout-first",
        "-l",
        action="store_true",
        default=None,
        help="Open a logout page first",
    )
    logout_first_group.add_argument(
        "--no-logout-first",
        dest="logout_first",
        action="store_false",
        help="Do not open a logout page first",
    )

    args = parser.parse_args()

    if args.open_url is None:
        args.open_url = True

    logout_first_from_env = False
    if args.logout_first is None:
        args.logout_first = os.environ.get("AWS_CONSOLE_LOGOUT_FIRST", "").lower() in [
            "true",
            "1",
        ]
        logout_first_from_env = True

    if args.logout_first and not args.open_url:
        if logout_first_from_env:
            logout_first_value = os.environ["AWS_CONSOLE_LOGOUT_FIRST"]
            raise parser.exit(
                f"AWS_CONSOLE_LOGOUT_FIRST={logout_first_value} requires --open"
            )
        else:
            raise parser.exit("--logout-first requires --open")

    session = boto3.Session(profile_name=args.profile)

    if not args.region:
        args.region = session.region_name or os.environ.get(
            "AWS_CONSOLE_DEFAULT_REGION"
        )
    if not args.destination_path:
        args.destination_path = session._session.get_scoped_config().get(
            "web_console_destination"
        ) or os.environ.get("AWS_CONSOLE_DEFAULT_DESTINATION")

    credentials = session.get_credentials()
    if not credentials:
        parser.exit("Could not find credentials")

    federation_endpoint = get_federation_endpoint(region=args.region)
    issuer = os.environ.get("AWS_CONSOLE_DEFAULT_ISSUER")
    destination = get_destination(
        path=args.destination_path,
        region=args.region,
        override_region_in_destination=args.override_region_in_destination,
    )

    launch_console(
        session=session,
        federation_endpoint=federation_endpoint,
        destination=destination,
        region=args.region,
        open_url=args.open_url,
        print_url=args.print_url,
        duration=args.duration,
        logout_first=args.logout_first,
        issuer=issuer,
    )


def launch_console(
    *,
    session: boto3.Session,
    federation_endpoint: str,
    destination: str,
    region: Optional[str] = None,
    open_url: Optional[bool] = None,
    print_url: Optional[bool] = None,
    duration: Optional[int] = None,
    logout_first: Optional[bool] = None,
    issuer: Optional[str] = None,
):
    if not issuer:
        issuer = "aws_console_launcher.py"

    read_only_credentials = session.get_credentials().get_frozen_credentials()

    session_data = {
        "sessionId": "**********"
        "sessionKey": "**********"
        "sessionToken": "**********"
    }

    get_signin_token_payload = "**********"
        "Action": "**********"
        "Session": json.dumps(session_data),
    }
    if duration is not None:
        get_signin_token_payload["SessionDuration"] = "**********"

    response = "**********"=get_signin_token_payload)

    if response.status_code != 200:
        print("Could not get signin token", file= "**********"
        print(response.status_code + "\n" + response.text, file=sys.stderr)
        sys.exit(2)

    token = "**********"

    get_login_url_params = {
        "Action": "login",
        "Issuer": issuer,
        "Destination": destination,
        "SigninToken": "**********"
    }

    request = requests.Request(
        method="GET", url=federation_endpoint, params=get_login_url_params
    )

    prepared_request = request.prepare()

    login_url = prepared_request.url

    if print_url:
        print(login_url)

    if open_url:
        if logout_first:
            logout_url = get_logout_url(region=region)
            webbrowser.open(
                logout_url, autoraise=False
            )  # &redirect_uri=https://aws.amazon.com

        webbrowser.open(login_url)


if __name__ == "__main__":
    main()
