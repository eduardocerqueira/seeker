#date: 2023-05-03T16:58:22Z
#url: https://api.github.com/gists/a8e5cb26f4d905a50e8ba0fc3552fd6b
#owner: https://api.github.com/users/jvarghese01

import qrcode
import base64
# generate qrcode for string

def get_qrcode(string):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(string)
    qr.make(fit=True)
    img = qr.make_image()

    return img


def lambda_handler(event, context):

    img = get_qrcode("hello world")
    img.save("/tmp/qrcode.png")
    image = open("/tmp/qrcode.png", "rb").read()
    
    return {
        "statusCode": 200,
        'body': base64.b64encode(image).decode('utf-8'),
        'isBase64Encoded': True,
        "headers": {
            "Content-Type": "image/png"
        }
    }
    
    
   