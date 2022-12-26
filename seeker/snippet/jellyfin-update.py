#date: 2022-12-26T16:20:32Z
#url: https://api.github.com/gists/b39a7ecfbeff8049eff08e1a0fdaf7be
#owner: https://api.github.com/users/cjerrington

from urllib.request import urlopen
import json

# Setup of variables:
# Update server and port to your local values
server = "localhost"
port = 8096
githubAPI = "https://api.github.com/repos/jellyfin/jellyfin/releases/latest"

localresp = urlopen(f"http://{server}:{port}/System/Info/Public")
myVersion = json.loads(localresp.read())

remoteresp = urlopen("https://api.github.com/repos/jellyfin/jellyfin/releases/latest")
remoteversion = json.loads(remoteresp.read())

if myVersion['Version'] == remoteversion['name']:
    print(f"{myVersion['Version']} is the latest version")
else:
    print(f"You are running {myVersion['Version']} and {remoteversion['name']} is the latest.")
    print(f"You can download it from {remoteversion['html_url']}")
    print(f"{remoteversion['body']}")
