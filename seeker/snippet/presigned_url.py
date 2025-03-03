#date: 2025-03-03T17:02:00Z
#url: https://api.github.com/gists/74e601913dc6480c7229c138b629da8b
#owner: https://api.github.com/users/drorasaf

import binascii
import collections
import datetime as dt
import hashlib
import os
from http import HTTPStatus
from urllib.parse import quote

import functions_framework
import google.cloud.logging
import six
from google.oauth2 import service_account
from config import settings

SERVICE_ACCOUNT_KEY_SECRET = "**********"
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()


def sign_url(
    service_account_file,
    bucket_name,
    object_name,
    expiration=86400,  # default is a day
    http_method="GET",
):
    """Generates a v4 signed URL for downloading a blob. Must use GCP credentials from service account file."""

    if expiration > 604800:
        logging.info("Expiration Time can't be longer than 604800 seconds (7 days).")
        raise ValueError("Expiration is longer than 7 days.")

    logging.info(
        f"Generating signed url. bucket_name: {bucket_name} object_name: {object_name}"
    )

    escaped_object_name = quote(six.ensure_binary(object_name), safe=b"/~")
    canonical_uri = f"/{escaped_object_name}"

    datetime_now = dt.datetime.now(tz=dt.timezone.utc)
    request_timestamp = datetime_now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = datetime_now.strftime("%Y%m%d")

    google_credentials = service_account.Credentials.from_service_account_file(
        service_account_file
    )
    client_email = google_credentials.service_account_email
    credential_scope = f"{datestamp}/auto/storage/goog4_request"
    credential = f"{client_email}/{credential_scope}"

    headers = {}
    host = f"{bucket_name}.storage.googleapis.com"
    headers["host"] = host

    canonical_headers = ""
    ordered_headers = collections.OrderedDict(sorted(headers.items()))
    for k, v in ordered_headers.items():
        lower_k = str(k).lower()
        strip_v = str(v).lower()
        canonical_headers += f"{lower_k}:{strip_v}\n"

    signed_headers = ""
    for k, _ in ordered_headers.items():
        lower_k = str(k).lower()
        signed_headers += f"{lower_k};"
    signed_headers = signed_headers[:-1]  # remove trailing ';'

    query_parameters = dict()
    query_parameters["X-Goog-Algorithm"] = "GOOG4-RSA-SHA256"
    query_parameters["X-Goog-Credential"] = credential
    query_parameters["X-Goog-Date"] = request_timestamp
    query_parameters["X-Goog-Expires"] = expiration
    query_parameters["X-Goog-SignedHeaders"] = signed_headers

    canonical_query_string = ""
    ordered_query_parameters = collections.OrderedDict(sorted(query_parameters.items()))
    for k, v in ordered_query_parameters.items():
        encoded_k = quote(str(k), safe="")
        encoded_v = quote(str(v), safe="")
        canonical_query_string += f"{encoded_k}={encoded_v}&"
    canonical_query_string = canonical_query_string[:-1]  # remove trailing '&'

    canonical_request = "\n".join(
        [
            http_method,
            canonical_uri,
            canonical_query_string,
            canonical_headers,
            signed_headers,
            "UNSIGNED-PAYLOAD",
        ]
    )

    canonical_request_hash = hashlib.sha256(canonical_request.encode()).hexdigest()

    string_to_sign = "\n".join(
        [
            "GOOG4-RSA-SHA256",
            request_timestamp,
            credential_scope,
            canonical_request_hash,
        ]
    )

    # signer.sign() signs using RSA-SHA256 with PKCS1v15 padding
    signature = binascii.hexlify(
        google_credentials.signer.sign(string_to_sign)
    ).decode()

    scheme_and_host = "{}://{}".format("https", host)
    signed_url = "{}{}?{}&x-goog-signature={}".format(
        scheme_and_host, canonical_uri, canonical_query_string, signature
    )

    logging.info(f"url: {signed_url}")
    return signed_url


@functions_framework.http
def generate_signed_url(request):
    """Cloud Function to be triggered to generate signed urls."""
    sa_filename = settings.service_account_temp_file
    logging.info(request)
    params = request.get_json()
    path = params["path"]
    # download signed url
    with open(sa_filename, "w", encoding="utf-8") as sa_file:
        sa_file.write(SERVICE_ACCOUNT_KEY_SECRET)
    try:
        url = sign_url(
            service_account_file=sa_filename,
            bucket_name=settings.bucket_name,
            object_name=path,
        )
    except Exception:
        raise
    finally:
        os.remove(sa_filename)
    return url, HTTPStatus.OK   os.remove(sa_filename)
    return url, HTTPStatus.OK