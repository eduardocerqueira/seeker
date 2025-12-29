#date: 2025-12-29T16:51:21Z
#url: https://api.github.com/gists/889ebef5bfbdc746675bfdd852b6171e
#owner: https://api.github.com/users/l1asis

#!/usr/bin/python3

# Schedule a live stream on your YouTube channel.
# Sample usage:
#   python create_broadcast.py --broadcast_title="Hi all!"

import sys
import json
import argparse
from datetime import datetime, timedelta, timezone

# Google API imports
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# The CLIENT_SECRETS_FILE variable specifies the name of a file that contains
# the OAuth 2.0 information for this application, including its client_id and
# client_secret. You can acquire an OAuth 2.0 client ID and client secret from
# the {{ Google Cloud Console }} at
#   https://console.cloud.google.com/apis/
# Please ensure that you have enabled the YouTube Data API for your project.
# For more information about using OAuth2 to access the YouTube Data API, see:
#   https://developers.google.com/youtube/v3/guides/authentication
 "**********"# "**********"  "**********"F "**********"o "**********"r "**********"  "**********"m "**********"o "**********"r "**********"e "**********"  "**********"i "**********"n "**********"f "**********"o "**********"r "**********"m "**********"a "**********"t "**********"i "**********"o "**********"n "**********"  "**********"a "**********"b "**********"o "**********"u "**********"t "**********"  "**********"t "**********"h "**********"e "**********"  "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"s "**********". "**********"j "**********"s "**********"o "**********"n "**********"  "**********"f "**********"i "**********"l "**********"e "**********"  "**********"f "**********"o "**********"r "**********"m "**********"a "**********"t "**********", "**********"  "**********"s "**********"e "**********"e "**********": "**********"
#   https: "**********"
CLIENT_SECRETS_FILE = "**********"

# The CREDENTIALS_FILE variable specifies the name of a file that contains
# the refresh_token which allows you to have short-lived access tokens without
# having to collect credentials every time one expires.
CREDENTIALS_FILE = "credentials.json"

# This OAuth 2.0 access scope allows for full read/write access to the
# authenticated user's account.
YOUTUBE_READ_WRITE_SCOPE = "https://www.googleapis.com/auth/youtube"
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

SCOPES = [YOUTUBE_READ_WRITE_SCOPE]

def get_saved_credentials(filename=CREDENTIALS_FILE):
    """ Read in any saved OAuth data/tokens """
    fileData = {}
    try:
        with open(filename, 'r') as file:
            fileData: dict = json.load(file)
    except FileNotFoundError:
        return None
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"f "**********"i "**********"l "**********"e "**********"D "**********"a "**********"t "**********"a "**********"  "**********"a "**********"n "**********"d "**********"  "**********"' "**********"r "**********"e "**********"f "**********"r "**********"e "**********"s "**********"h "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"' "**********"  "**********"i "**********"n "**********"  "**********"f "**********"i "**********"l "**********"e "**********"D "**********"a "**********"t "**********"a "**********"  "**********"a "**********"n "**********"d "**********"  "**********"' "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********"_ "**********"i "**********"d "**********"' "**********"  "**********"i "**********"n "**********"  "**********"f "**********"i "**********"l "**********"e "**********"D "**********"a "**********"t "**********"a "**********"  "**********"a "**********"n "**********"d "**********"  "**********"' "**********"c "**********"l "**********"i "**********"e "**********"n "**********"t "**********"_ "**********"s "**********"e "**********"c "**********"r "**********"e "**********"t "**********"' "**********"  "**********"i "**********"n "**********"  "**********"f "**********"i "**********"l "**********"e "**********"D "**********"a "**********"t "**********"a "**********": "**********"
        return Credentials(**fileData)
    return None

def store_creds(credentials, filename=CREDENTIALS_FILE):
    """ Save refresh_token with other credentials in the file """
    if not isinstance(credentials, Credentials):
        return
    fileData = {'refresh_token': "**********"
                'token': "**********"
                'client_id': credentials.client_id,
                'client_secret': "**********"
                'token_uri': "**********"
    with open(filename, 'w') as file:
        json.dump(fileData, file, indent=" "*4)
    print(f'[INFO] Credentials serialized to {filename}.')

 "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"c "**********"r "**********"e "**********"d "**********"e "**********"n "**********"t "**********"i "**********"a "**********"l "**********"s "**********"_ "**********"v "**********"i "**********"a "**********"_ "**********"o "**********"a "**********"u "**********"t "**********"h "**********"( "**********"f "**********"i "**********"l "**********"e "**********"n "**********"a "**********"m "**********"e "**********"= "**********"C "**********"L "**********"I "**********"E "**********"N "**********"T "**********"_ "**********"S "**********"E "**********"C "**********"R "**********"E "**********"T "**********"S "**********"_ "**********"F "**********"I "**********"L "**********"E "**********", "**********"  "**********"s "**********"c "**********"o "**********"p "**********"e "**********"s "**********"= "**********"S "**********"C "**********"O "**********"P "**********"E "**********"S "**********", "**********"  "**********"s "**********"a "**********"v "**********"e "**********"D "**********"a "**********"t "**********"a "**********"= "**********"T "**********"r "**********"u "**********"e "**********") "**********"  "**********"- "**********"> "**********"  "**********"C "**********"r "**********"e "**********"d "**********"e "**********"n "**********"t "**********"i "**********"a "**********"l "**********"s "**********": "**********"
    """ Use data in the given filename to get oauth data """
    iaflow = "**********"
    iaflow.run_local_server()
    if saveData:
        store_creds(iaflow.credentials)
    return iaflow.credentials

def get_service(credentials, service=YOUTUBE_API_SERVICE_NAME, version=YOUTUBE_API_VERSION):
    """ Construct a Resource for interacting with an YouTube API. """
    return build(service, version, credentials=credentials)

def get_or_insert_stream(youtube, options):
    """ Show all your RTMP/HLS Keys (a-ka Live Streams) and get one if 'stream-name' (a-ka RTMP key) option is set """
    request = youtube.liveStreams().list(
        part="snippet,cdn,contentDetails,status",
        maxResults=99,
        mine=True
    )
    response = request.execute()
    streams = response["items"]
    if options.stream_key:
        for stream in streams:
            streamId = stream["id"]
            rtmpKey = stream["cdn"]["ingestionInfo"]["streamName"]
            ingestionAddress = stream["cdn"]["ingestionInfo"]["ingestionAddress"]
            rtmpAddress = f"{ingestionAddress}/{rtmpKey}"
            if options.stream_key == rtmpKey:
                print(f"[INFO] Found the user-defined RTMP key.")
                print(f"[+] RTMP ADDRESS: {rtmpAddress}")
                return streamId
        else:
            return insert_stream(youtube, args)
    else:
        return insert_stream(youtube, args)


def insert_stream(youtube, options):
    """ Create a liveStream resource and set its title, description, format, and ingestion type.\n
        This resource describes the content that you are transmitting to YouTube. """
    request = youtube.liveStreams().insert(
        part="snippet,cdn",
        body={
          "cdn": {
            "format": options.stream_format, # flv
            "resolution": options.stream_resolution, # 1080p
            "frameRate": options.stream_framerate, # 60fps
            "ingestionType": options.stream_ingestiontype # rtmp
          },
          "snippet": {
            "title": options.stream_title,
            "description": options.stream_description,
          }
        }
    )
    response = request.execute()
    streamIngestionAddress = response["cdn"]["ingestionInfo"]["ingestionAddress"]
    streamKey = response["cdn"]["ingestionInfo"]["streamName"]
    rtmpAddress = f"{streamIngestionAddress}/{streamKey}"
    print(f"[INFO] Did not find the user-defined RTMP key. Creating a new one instead:")
    print(f"[+] RTMP ADDRESS: {rtmpAddress}")
    print("[+] Stream '{0}' with title '{1}' was inserted.".format(response["id"], response["snippet"]["title"]))
    return response["id"]

def insert_broadcast(youtube, options):
    """ Create a liveBroadcast resource and set its title, description,
        scheduled start time, scheduled end time, and privacy status. """
    request = youtube.liveBroadcasts().insert(
        part="snippet,status,contentDetails",
        body={
          "snippet": {
            "title": options.broadcast_title,
            "description": options.broadcast_description,
            "scheduledStartTime": options.start_time,
            "scheduledEndTime": options.end_time
          },
          "status": {
            "privacyStatus": options.privacy_status,
            "selfDeclaredMadeForKids": False
          },
          "contentDetails": {
              "enableAutoStart": True
          }
        }
    )
    response = request.execute()
    print("[+] Broadcast '{0}' with title '{1}' was published at '{2}'.".format(response["id"], response["snippet"]["title"], response["snippet"]["publishedAt"]))
    return response["id"]

def bind_broadcast(youtube, broadcast_id, stream_id):
    """ Bind the broadcast to the video stream. By doing so, you link the video that
        you will transmit to YouTube to the broadcast that the video is for. """
    request = youtube.liveBroadcasts().bind(
        part="id,contentDetails",
        id=broadcast_id,
        streamId=stream_id
    )
    response = request.execute()
    print("[+] Broadcast '{0}' was bound to stream '{1}'.".format(
        response["id"], response["contentDetails"]["boundStreamId"]))
    print("[+] BROADCAST ID: {0}".format(response["id"]))

def end_broadcast(youtube, args):
    """ End the broadcast so that YouTube stops transmitting video. """
    request = youtube.liveBroadcasts().transition(
        broadcastStatus="complete",
        id=args.broadcast_id,
        part="id,status"
    )
    response = request.execute()
    print("[+] Broadcast '{0}' has been successfully completed.".format(response["id"]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='Broadcast Binder')

    # --- First separate group just to order the arguments in help message
    group_create = parser.add_argument_group("Create & Schedule YouTube Broadcast Arguments")
    group_create.add_argument("--create", action="store_true", help="Argument to schedule broadcast")
    group_create.add_argument("--broadcast-title", dest="broadcast_title", help="Broadcast title", default="New Broadcast")
    group_create.add_argument("--broadcast-desc", dest="broadcast_description", help="Broadcast description", default=" ")
    group_create.add_argument("--privacy-status", dest="privacy_status", help="Broadcast privacy status", default="public")
    group_create.add_argument("--start-time", dest="start_time", help="Scheduled start time",
                        default=(datetime.now(timezone.utc).astimezone() + timedelta(seconds=30)).isoformat())
    group_create.add_argument("--end-time", dest="end_time", help="Scheduled end time",
                        default=(datetime.now(timezone.utc).astimezone() + timedelta(hours=6)).isoformat())
    group_create.add_argument("--stream-key", dest="stream_key", help="Your preferred RTMP key (OPTIONAL)", default=None)
    group_create.add_argument("--stream-title", dest="stream_title", help="Stream title", default="New Stream")
    group_create.add_argument("--stream-desc", dest="stream_description", help="Stream description", default=" ")
    group_create.add_argument("--stream-format", dest="stream_format", help="Stream format", default="flv")
    group_create.add_argument("--stream-resolution", dest="stream_resolution", help="Stream resolution", default="1080p")
    group_create.add_argument("--stream-framerate", dest="stream_framerate", help="Stream framerate", default="60fps")
    group_create.add_argument("--stream-ingestiontype", dest="stream_ingestiontype", help="Stream ingestion type", default="rtmp")

    # --- Seconds separate group just to order the arguments in help message
    group_terminate = parser.add_argument_group("End YouTube Broadcast Arguments")
    group_terminate.add_argument("--terminate", action="store_true", help="Argument to end broadcast")
    group_terminate.add_argument("--broadcast-id", dest="broadcast_id", help="Broadcast ID", default=None)

    group_other = parser.add_argument_group("Other")
    group_other.add_argument("--working-directory", dest="working_directory", help="Your main directory with all scripts and other stuff", default=None)

    args = parser.parse_args()

    # Setting the additional folder in the PATH
    if args.working_directory:
        sys.path.append(args.working_directory)

    credentials = get_saved_credentials()
    if not credentials:
        credentials = get_credentials_via_oauth()
    youtube = get_service(credentials)

    try:
        if not args.create and not args.terminate:
            print("[ERR] You should choose to either create or cancel the broadcast.\n[HELP] -h argument")
        elif args.create and args.terminate:
            print("[ERR] You cannot create and terminate the broadcast at the same time.\n[HELP] -h argument")
        elif args.create:
            broadcast_id = insert_broadcast(youtube, args)
            stream_id = get_or_insert_stream(youtube, args)
            bind_broadcast(youtube, broadcast_id, stream_id)
        elif args.terminate:
            if args.broadcast_id:
                end_broadcast(youtube, args)
            else:
                print("[ERR] You have not specified the broadcast ID.\n[HELP] -h argument")
            
    except HttpError as error:
        print("[ERR] An HTTP error {0} occured:\n{1}".format(error.status_code, error.content))
