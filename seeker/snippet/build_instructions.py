#date: 2022-06-13T16:57:29Z
#url: https://api.github.com/gists/8f96edcdd6f2efbec76a8410dfc102d9
#owner: https://api.github.com/users/koaning

import base64
import pathlib
from jinja2 import Environment, FileSystemLoader, select_autoescape

env = Environment(
    loader=FileSystemLoader("images"),
    autoescape=select_autoescape()
)

template = env.get_template("instructions.template")

with open("images/img4.png", "rb") as image_file:
    enc_noa = base64.b64encode(image_file.read())

with open("images/img5.png", "rb") as image_file:
    enc_sok = base64.b64encode(image_file.read())

rendered = template.render(enc_noa=enc_noa.decode('utf-8'), enc_sok=enc_sok.decode('utf-8'))
pathlib.Path("instructions.html").write_text(rendered)
