#date: 2022-07-08T17:16:50Z
#url: https://api.github.com/gists/187691ce3d690da6c7f6fb24d42a6469
#owner: https://api.github.com/users/michelssousa

from s3 import AwsS3UploadClass
from config import id_key
from config import secret_key
from config import bucket_name
from flask import Flask
from flask import jsonify
from flask import request
import requests

app = Flask(__name__)


@app.route('/upload_file', methods=["POST"])
def upload_file():
  if request.method == "POST":
        file = None
        if "file" in request.files:
            file = request.files['file']
        else:
            return jsonify(error="requires file")
                # import s3 upload class

        s3 = AwsS3UploadClass(id_key,secret_key,bucket_name)
        # save file
        key = "file_name"
        file.save(key)
        # generate presgined post url
        response = s3.create_presigned_post(key)
        if response is None:
            return jsonify(error="key cannot None")

        files = [
            ('file', open(key, 'rb'))
        ]
        upload_response = requests.post(response['url'], data=response['fields'], files=files)

        if upload_response.status_code == 204:
            # remove file
            os.remove(key)
        return jsonify("file successfully uploaded to s3")
      
if __name__ == '__main__':
    app.run()