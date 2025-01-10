#date: 2025-01-10T17:09:09Z
#url: https://api.github.com/gists/6e0653089bbf05ab59815f4562e7205e
#owner: https://api.github.com/users/memo-py

from flask import Flask, request, jsonify
from pytube import YouTube
import os

app = Flask(__name__)

@app.route('/download', methods=['POST'])
def download_video():
    try:
        data = request.json
        url = data.get('url')
        path = data.get('path', './downloads')

        # Create the downloads folder if it doesn't exist
        if not os.path.exists(path):
            os.makedirs(path)

        yt = YouTube(url)
        video_stream = yt.streams.filter(progressive=True, file_extension='mp4').first()

        if video_stream:
            video_stream.download(path)
            return jsonify({"message": f"Downloaded '{yt.title}' successfully!", "status": "success"}), 200
        else:
            return jsonify({"message": "No available video streams found.", "status": "error"}), 400

    except Exception as e:
        return jsonify({"message": str(e), "status": "error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
