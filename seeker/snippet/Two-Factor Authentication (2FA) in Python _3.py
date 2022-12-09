#date: 2022-12-09T17:10:26Z
#url: https://api.github.com/gists/ce12b79f19f1fd999fff058b93746b03
#owner: https://api.github.com/users/BOBAZIZ

import time
import pyotp
import qrcode

key = "NewSuperCode"

uri = pyotp.totp.TOTP(key).provisioning_uri(name="ShefiAziz",
                                            issuer_name="NeuralNine App")

print(uri)

qrcode.make(uri).save("totp.png")