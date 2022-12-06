#date: 2022-12-06T16:45:44Z
#url: https://api.github.com/gists/c8867e3d3fdb907480e54e305c80409f
#owner: https://api.github.com/users/MarkusKirschner

def check_jwt():
    from authlib.jose import jwt
    token = "**********"
    public_key = "...insert base64-public-RSA-key..."
    key = '-----BEGIN PUBLIC KEY-----\n' + public_key + '\n-----END PUBLIC KEY-----'
    key_binary = key.encode('ascii')

    try:
        claims = "**********"
        claims.validate()
    except Exception as e:
        abort(f"Invalid token: "**********"
    logging.debug(f"Token claims: "**********"n claims: {claims}")