#date: 2024-06-06T16:43:23Z
#url: https://api.github.com/gists/322c1d2f33ea7b8e83664f6b2293336c
#owner: https://api.github.com/users/alessandromasone

import os
import pickle
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import time
import random
import logging
import datetime


# Definisci le costanti
CLIENT_SECRETS_FILE = "**********"
SCOPES = ['https://www.googleapis.com/auth/youtube.upload']
API_SERVICE_NAME = 'youtube'
API_VERSION = 'v3'
TOKEN_PICKLE_FILE = "**********"

# Configura il sistema di logging
logging.basicConfig(filename='upload_log.txt', level=logging.INFO)
logger = logging.getLogger()

# Formattatore per includere l'orario nei messaggi di log
formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler('upload_log.txt')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Funzione per la stampa dei messaggi
def print_message(message):
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_message = f"{timestamp} - {message}"
    print(formatted_message)
    logger.info(formatted_message)

class UploadShorts:
    UPLOAD_RECORDS_FILE = "upload_records.txt"

    @staticmethod
    def choose_random_video(directory):
        files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
        available_files = [f for f in files if f not in open(UploadShorts.UPLOAD_RECORDS_FILE).read()]
        if available_files:
            return os.path.join(directory, random.choice(available_files))
        else:
            return None

    @staticmethod
    def save_upload(file_name):
        try:
            with open(UploadShorts.UPLOAD_RECORDS_FILE, "a") as file:
                file.write(file_name + "\n")
            print_message("Upload record saved successfully.")
        except Exception as e:
            print_message(f"Error saving upload record: {str(e)}")

    @staticmethod
    def get_number():
        try:
            with open(UploadShorts.UPLOAD_RECORDS_FILE, 'r') as file:
                return sum(1 for line in file)
        except Exception as e:
            print_message(f"Error getting number of video: {str(e)}")
            return None



def get_authenticated_service():
    credentials = None

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"o "**********"s "**********". "**********"p "**********"a "**********"t "**********"h "**********". "**********"e "**********"x "**********"i "**********"s "**********"t "**********"s "**********"( "**********"T "**********"O "**********"K "**********"E "**********"N "**********"_ "**********"P "**********"I "**********"C "**********"K "**********"L "**********"E "**********"_ "**********"F "**********"I "**********"L "**********"E "**********") "**********": "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"T "**********"O "**********"K "**********"E "**********"N "**********"_ "**********"P "**********"I "**********"C "**********"K "**********"L "**********"E "**********"_ "**********"F "**********"I "**********"L "**********"E "**********", "**********"  "**********"' "**********"r "**********"b "**********"' "**********") "**********"  "**********"a "**********"s "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            credentials = "**********"

    if not credentials or not credentials.valid:
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"c "**********"r "**********"e "**********"d "**********"e "**********"n "**********"t "**********"i "**********"a "**********"l "**********"s "**********"  "**********"a "**********"n "**********"d "**********"  "**********"c "**********"r "**********"e "**********"d "**********"e "**********"n "**********"t "**********"i "**********"a "**********"l "**********"s "**********". "**********"e "**********"x "**********"p "**********"i "**********"r "**********"e "**********"d "**********"  "**********"a "**********"n "**********"d "**********"  "**********"c "**********"r "**********"e "**********"d "**********"e "**********"n "**********"t "**********"i "**********"a "**********"l "**********"s "**********". "**********"r "**********"e "**********"f "**********"r "**********"e "**********"s "**********"h "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            credentials.refresh(Request())
        else:
            print_message("User action requested")
            flow = "**********"
            credentials = flow.run_local_server(port=0)

        # Salva le credenziali
 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"T "**********"O "**********"K "**********"E "**********"N "**********"_ "**********"P "**********"I "**********"C "**********"K "**********"L "**********"E "**********"_ "**********"F "**********"I "**********"L "**********"E "**********", "**********"  "**********"' "**********"w "**********"b "**********"' "**********") "**********"  "**********"a "**********"s "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
            pickle.dump(credentials, token)

    return build(API_SERVICE_NAME, API_VERSION, credentials=credentials)



def upload_video(youtube, file, title, description, privacy='private'):
    try:
        request = youtube.videos().insert(
            part="snippet,status",
            body={
                "snippet": {
                    "title": title,
                    "description": description
                },
                "status": {
                    "privacyStatus": privacy
                }
            },
            media_body=MediaFileUpload(file)
        )
        response = request.execute()

        print_message(f"Video uploaded successfully! \n Title: {title} \n Description: {description} \n Privacy Status: {privacy}")
        UploadShorts.save_upload(file)
    except HttpError as e:
        error_message = json.loads(e.content)['error']['message']
        print_message(f"Error uploading video: {error_message}")
    except Exception as e:
        print_message(f"Error uploading video: {str(e)}")



def main():

    while True:
        print_message("Prossimo video\n")
        youtube = get_authenticated_service()
        upload_video(youtube, UploadShorts.choose_random_video('shorts'), f"Tarkov clip #{UploadShorts.get_number()}", f"Tarkov clip #{UploadShorts.get_number()}", 'public')


if __name__ == '__main__':
    main()
