#date: 2025-10-02T16:52:13Z
#url: https://api.github.com/gists/05405a99d058b341299063286c011f3f
#owner: https://api.github.com/users/jdgregson

#!/usr/bin/env python3
from flask import Flask, request, Response, jsonify
import requests
import json
import datetime
import logging
import re
from jsonschema import validate, ValidationError
from auth_db import AuthDB

app = Flask(__name__)
COMFYUI_HOST = "127.0.0.1:8188"
COMFYUI_URL = f"http://{COMFYUI_HOST}"
auth_db = AuthDB()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

with open('schemas/workflow.json', 'r') as f:
    WORKFLOW_SCHEMA = json.load(f)
with open('schemas/image_request.json', 'r') as f:
    IMAGE_REQUEST_SCHEMA = json.load(f)


def validate_request(instance, schema):
    try:
        validate(instance=instance, schema=schema)
        return None
    except ValidationError as e:
        return jsonify({"Validation error": str(e)}), 400
    except Exception as e:
        return jsonify({"Unexpected error": str(e)}), 400


@app.route('/<path:path>', methods=['GET', 'POST', 'PUT', 'DELETE'])
def proxy(path):

    client_token = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
        return jsonify({"error": "**********"

    if request.method == 'POST' and path == 'prompt':
        payload = request.get_json(force=True)
        validation_result = validate_request(instance=payload, schema=WORKFLOW_SCHEMA)
        if validation_result:
            return validation_result

        # Extract and register image ID for this client
        image_id = payload['prompt']['9']['inputs']['filename_prefix']
        auth_db.register_image(image_id, client_token)

    elif request.method == 'GET' and path == 'view':
        validation_result = validate_request(instance=dict(request.args), schema=IMAGE_REQUEST_SCHEMA)
        if validation_result:
            return validation_result

        # Extract image ID from filename and verify ownership
        filename = request.args.get('filename', '')
        image_id = re.match(r'^([0-9a-f-]+)_', filename)
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"n "**********"o "**********"t "**********"  "**********"i "**********"m "**********"a "**********"g "**********"e "**********"_ "**********"i "**********"d "**********"  "**********"o "**********"r "**********"  "**********"n "**********"o "**********"t "**********"  "**********"a "**********"u "**********"t "**********"h "**********"_ "**********"d "**********"b "**********". "**********"v "**********"e "**********"r "**********"i "**********"f "**********"y "**********"_ "**********"a "**********"c "**********"c "**********"e "**********"s "**********"s "**********"( "**********"i "**********"m "**********"a "**********"g "**********"e "**********"_ "**********"i "**********"d "**********". "**********"g "**********"r "**********"o "**********"u "**********"p "**********"( "**********"1 "**********") "**********", "**********"  "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********") "**********": "**********"
            return jsonify({"error": "Access denied"}), 403

    else:
        return jsonify({"error": "Not found"}), 404

    log_data = {
        'timestamp': datetime.datetime.now().isoformat(),
        'method': request.method,
        'path': path,
        'query_params': dict(request.args),
        'headers': dict(request.headers)
    }
    logger.info(f"Proxying request: {json.dumps(log_data)}")

    comfyui_response = requests.request(
        method=request.method,
        url=f"{COMFYUI_URL}/{path}",
        headers={},
        data=request.get_data(),
        params=request.args,
        allow_redirects=False
    )

    logger.info(f"Proxied response: {comfyui_response.status_code}")
    return Response(comfyui_response.content, comfyui_response.status_code, comfyui_response.headers.items())


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8189, debug=False)
