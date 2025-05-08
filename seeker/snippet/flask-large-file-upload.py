#date: 2025-05-08T16:39:08Z
#url: https://api.github.com/gists/0e59751db10088c4fd69fc87033ec3cd
#owner: https://api.github.com/users/AalbatrossGuy

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method =="POST":
        for file in request.files: # Get the file using flask.request from your <input> tag 
            if file.startswith('file'):
                get_file = request.files.get(file) # When uploading multiple files, use the loop
                with open(f"{os.getenv('ROOT_DIRECTORY')}/{get_file.filename}", 'wb') as file_binary: # wb will create the file if it doesn't exist and write the chunk
                    for file_chunk in get_file.stream: # Get the file in a stream reader
                        file_binary.write(file_chunk)
    return render_template("upload.html")