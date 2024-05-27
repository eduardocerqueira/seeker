#date: 2024-05-27T17:06:34Z
#url: https://api.github.com/gists/882b75273a004ee4d4312e6b6c9ca7c5
#owner: https://api.github.com/users/m-mahdi-sangtarash

import requests

def download_file_from_google_drive(id, destination):
 "**********"  "**********"  "**********"  "**********"  "**********"d "**********"e "**********"f "**********"  "**********"g "**********"e "**********"t "**********"_ "**********"c "**********"o "**********"n "**********"f "**********"i "**********"r "**********"m "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"( "**********"r "**********"e "**********"s "**********"p "**********"o "**********"n "**********"s "**********"e "**********") "**********": "**********"
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********": "**********"
        params = { 'id' : "**********": token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


if __name__ == "__main__":
    import sys
    if len(sys.argv) is not 3:
        print("Usage: python google_drive.py drive_file_id destination_file_path")
    else:
        # TAKE ID FROM SHAREABLE LINK
        file_id = sys.argv[1]
        # DESTINATION FILE ON YOUR DISK
        destination = sys.argv[2]
        download_file_from_google_drive(file_id, destination)