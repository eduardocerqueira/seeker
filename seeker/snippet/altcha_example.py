#date: 2025-03-17T16:55:55Z
#url: https://api.github.com/gists/1fdb30eed5dfcd84d2a1ec28e00b60f8
#owner: https://api.github.com/users/admin-else

import base64
from flask import Flask, request, render_template, url_for, jsonify
import datetime
import json
from altcha import ChallengeOptions, create_challenge, verify_solution

app = Flask(__name__)

HMAC_KEY = "**********"

@app.route("/")
def index():
    # Render the page containing the ALTCHA widget.
    return render_template("captcha.html")

@app.route("/challenge", methods=["GET"])
def get_challenge():
    options = ChallengeOptions(
        expires=datetime.datetime.now() + datetime.timedelta(minutes=10),
        max_number=100000,
        hmac_key=HMAC_KEY,
    )
    captcha = create_challenge(options)
    return jsonify({
        "algorithm": captcha.algorithm,
        "challenge": captcha.challenge,
        "salt": captcha.salt,
        "signature": captcha.signature,
    })

@app.route("/verify", methods=["POST"])
def verify():
    altcha_payload = request.form.get("altcha")
    print(altcha_payload)
    if not altcha_payload:
        return "Missing altcha payload", 400
    try:
        payload = json.loads(base64.decodebytes(altcha_payload.encode()))
    except json.JSONDecodeError:
        return "Invalid altcha payload", 400

    ok, err = verify_solution(payload, HMAC_KEY, check_expires=True)
    if err:
        message = f"Error: {err}"
    elif ok:
        message = "Solution verified!"
    else:
        message = "Invalid solution."
    return render_template("captcha.html", message=message)

if __name__ == "__main__":
    app.run(debug=True)
rue)
