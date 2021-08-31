#date: 2021-08-31T03:24:30Z
#url: https://api.github.com/gists/5187cd8f63ab771bd9bc36e06e9831fb
#owner: https://api.github.com/users/caihao6654

# -*- coding:utf-8 -*-
from flask import Flask, request, jsonify, render_template
import subprocess

app = Flask('test_api', static_folder='static', static_url_path='/static')

@app.route('/subprocess_run', methods=['post','get'])
def get_request_info():

    try:
        command = request.json.get("command")
        et = subprocess.run(str(command), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8",
                            timeout=20000)
        msg = et.stdout
        print(msg)

    except:
        return jsonify({"code": 10001, "msg": "执行失败"})

    return jsonify({"code": 10000, "msg": msg})


# app.run(port=8081, debug='true', host="0.0.0.0")

if __name__ == '__main__':
    app.run(port=8081, debug='true', host="0.0.0.0")

