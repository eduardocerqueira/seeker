#date: 2022-06-17T17:02:31Z
#url: https://api.github.com/gists/471bc58e08abcddfb2d6229a40b5aa9a
#owner: https://api.github.com/users/katrbhach

import requests
import json
from app import app
from .env_constants import EnvironmentConstants


@app.route("/health", methods=["GET"])
def health():
    """
    Test connection to kafka-rest-proxy and return appropriate status
    :return:
    """
    try:

        response = requests.get("{}".format(EnvironmentConstants.url),
                                auth=(EnvironmentConstants.username, EnvironmentConstants.password),
                                verify=False, timeout=30)
        response.raise_for_status()

        return app.response_class(response=json.dumps({"message": "success"}), status=200, mimetype='application/json')

    except Exception:

        app.logger.exception("failed to get response from kafka-rest-proxy")

        return app.response_class(response=json.dumps({"error": "failed to get response from kafka-rest-proxy"}),
                                  status=500,
                                  mimetype='application/json')
