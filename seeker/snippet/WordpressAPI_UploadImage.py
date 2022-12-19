#date: 2022-12-19T16:52:45Z
#url: https://api.github.com/gists/1b0afd43258b939395c854465f4a88b1
#owner: https://api.github.com/users/dhanushreddy291

import base64, requests, json

 "**********"d "**********"e "**********"f "**********"  "**********"h "**********"e "**********"a "**********"d "**********"e "**********"r "**********"( "**********"u "**********"s "**********"e "**********"r "**********", "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********") "**********": "**********"
    credentials = user + ': "**********"
    token = "**********"
    header_json = {'Authorization': "**********"
    return header_json

def upload_image_to_wordpress(file_path, url, header_json):
    media = {'file': open(file_path,"rb"),'caption': 'My great demo picture'}
    responce = requests.post(url + "wp-json/wp/v2/media", headers = header_json, files = media)
    print(responce.text)

hed = "**********"
upload_image_to_wordpress('C://Users//X//Desktop//X//X.png', 'https://XXX.xyz/XXX/',hed)