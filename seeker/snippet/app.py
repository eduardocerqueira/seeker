#date: 2022-07-06T16:50:45Z
#url: https://api.github.com/gists/faee02b304b0444845703a3d130d928c
#owner: https://api.github.com/users/fmaida

import json
from flask import Flask, request


app = Flask(__name__)


def extract_jotform_data():
    output = {}
    form_data = request.form.to_dict()
    if form_data.get("rawRequest"):
        for key, value in json.loads(form_data["rawRequest"]).items():
            # Removes the "q<number>_" part from the key name
            # Instead of "q5_quantity" we want "quantity" as the key
            temp = key.split("_")
            new_key = key if len(temp) == 1 else "_".join(temp[1:])
            # Saves the item with the new key in the dictionary
            output[new_key] = value

    return output


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    jotform = extract_jotform_data()
    for key, value in jotform.items():
        print(f"{key}: {value}")
        if type(value) is dict:
            for subkey, subvalue in value.items():
                print(f" +------ {subkey}: {subvalue}")

    return "ok", 200


if __name__ == '__main__':
    app.run()
