#date: 2023-07-06T16:58:04Z
#url: https://api.github.com/gists/59d1f7e568c5e1c027de02490fcaaa6d
#owner: https://api.github.com/users/HarshitDawar55

# Importing the required libraries
from flask import Flask, request
import boto3
import os


# Defining the App with the Name
app = Flask(__name__)


# Creating a error handler for the Internal Server Error
@app.errorhandler(500)
def error_505(error):
    return "There is some problem with the application!"

# Creating a default route to translate the text into the destination language
@app.route("/", methods = ["POST"])
def translate():
    try:
        # Taking the required parameters as input from the API
        text = request.args.get("text")
        SourceLanguage = request.args.get("source_language")
        TargetLanguage = request.args.get("target_language")

        # Creating a translate client of AWS to translate the text
        language_translator = boto3.client(service_name = "translate", region_name = "ap-south-1",
                                            aws_access_key_id = "**********"
                                                    aws_secret_access_key = "**********"

        # Translating the text
        result = language_translator.translate_text(Text = text,
                                                    TargetLanguageCode = TargetLanguage,
                                                    SourceLanguageCode = SourceLanguage)

        # Returning the output
        return result.get('TranslatedText')
    except Exception as e:
        print(str(e))

# Running the app on port 80 & on any host to make it accessible through the container
app.run(host = "0.0.0.0", port = 80)run(host = "0.0.0.0", port = 80)