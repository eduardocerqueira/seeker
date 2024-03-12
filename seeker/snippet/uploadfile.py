#date: 2024-03-12T17:05:55Z
#url: https://api.github.com/gists/2257b5078839d5da8b2e21b80193e2d6
#owner: https://api.github.com/users/werdhaihai

from flask import Flask, request, redirect, url_for, render_template_string

app = Flask(__name__)

UPLOAD_FORM = '''
<!doctype html>
<title>File Upload</title>
<h1>Upload new File</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save file to ./uploads
            # don't forget to create this directory
            file.save(f"./uploads/{file.filename}")
            return 'File successfully uploaded'
    return render_template_string(UPLOAD_FORM)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
