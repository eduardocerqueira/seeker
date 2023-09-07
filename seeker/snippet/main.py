//date: 2023-09-07T16:52:34Z
//url: https://api.github.com/gists/2d9ca5773a1958cc7a4f5cc7b3faa8d4
//owner: https://api.github.com/users/vdparikh

from flask import Flask, request, jsonify

app = Flask(__name__)
key_value_store = {}


@app.route('/get', methods=['GET'])
def get():
    key = request.args.get('key')
    if key in key_value_store:
        return jsonify({key: key_value_store[key]}), 200
    else:
        return "Key not found", 404


@app.route('/set', methods=['POST'])
def set_key():
    data = request.get_json()
    if 'key' not in data or 'value' not in data:
        return "Invalid request, please provide 'key' and 'value'", 400
    key = data['key']
    value = data['value']
    key_value_store[key] = value
    return "Key-Value pair set successfully", 200


@app.route('/update', methods=['PUT'])
def update_key():
    data = request.get_json()
    if 'key' not in data or 'value' not in data:
        return "Invalid request, please provide 'key' and 'value'", 400
    key = data['key']
    if key in key_value_store:
        key_value_store[key] = data['value']
        return "Key-Value pair updated successfully", 200
    else:
        return "Key not found", 404


@app.route('/delete', methods=['DELETE'])
def delete_key():
    key = request.args.get('key')
    if key in key_value_store:
        del key_value_store[key]
        return "Key deleted successfully", 200
    else:
        return "Key not found", 404


@app.route('/list', methods=['GET'])
def list_keys():
    return jsonify(list(key_value_store.keys())), 200


if __name__ == '__main__':
    app.run(debug=True)
