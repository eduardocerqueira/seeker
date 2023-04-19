#date: 2023-04-19T17:02:40Z
#url: https://api.github.com/gists/c549d57be7a37eae88dbcf033c8bba84
#owner: https://api.github.com/users/ssoto

import qrcode
import uuid


def create_qr_code(uuid, qr_path):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(uuid)
    qr.make(fit=True)
    imagen_qr = qr.make_image(fill_color="black", back_color="white")
    imagen_qr.save(qr_path)


# Crea un UUID4
codigo = str(uuid.uuid4())
create_qr_code(codigo, './qr_image.png')
