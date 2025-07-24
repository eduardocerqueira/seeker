#date: 2025-07-24T16:52:38Z
#url: https://api.github.com/gists/bb33c30ac683dfab28079876216e44cf
#owner: https://api.github.com/users/mobinjavari

from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

@app.route('/', methods=['GET'])
def generate():
    data = {
        "model": "phi3:mini",
        "messages": [
            {
                "role": "system",
               "content": "You are python developer"
            },
            {
               "role": "user",
               "content": "how to print mobin in python"
            }
        ],
        "stream": False
    }

    url = 'http://localhost:11434/api/chat'

    try:
        response = requests.post(url, json=data)
        response.raise_for_status() 
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(response.json()), response.status_code

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
