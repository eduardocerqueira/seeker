#date: 2023-04-04T16:37:39Z
#url: https://api.github.com/gists/024ca3506e4f724ec5603e0125991eee
#owner: https://api.github.com/users/Ry-DS

# Python server with cors accepts incoming requests from the client and wakes pc
from flask import Flask, jsonify, request
from flask_cors import CORS
from wakeonlan import send_magic_packet

app = Flask(__name__)
CORS(app)
# run flask in prod mode
app.config['ENV'] = 'production'

@app.route('/wake', methods=['POST'])
def api():
    print("Waking PC")
    # wake pc with WOL
    send_magic_packet('2C:F0:5D:8B:??:??', ip_address='192.168.1.255')
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    send_magic_packet('2C:F0:5D:8B:??:??', ip_address='192.168.1.255')
    # app.run(host='server.ryan-s.me', port=99)