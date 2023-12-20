#date: 2023-12-20T16:52:45Z
#url: https://api.github.com/gists/55ff39220985c1f880721fb47196e971
#owner: https://api.github.com/users/yerffff

# Displays the currently playing media at FIP radio
# http://www.fipradio.fr/player

# You may want to update the number at the end of the URL
# if you use a specific "style" channel instead of the
# generic (main) channel, which is 7.

import requests
import time
import subprocess
import tempfile

IDs = {
  "FIP": {
    "ID": 7
  },
  "Rock": {
    "ID": 64
  },
  "Jazz": {
    "ID": 65
  },
  "Groove": {
    "ID": 66
  },
  "World": {
    "ID": 69
  },
  "Nouveautes": {
    "ID": 70
  },
  "Reggae": {
    "ID": 71
  },
  "Electro": {
    "ID": 74
  },
  "Metal": {
    "ID": 77
  },
  "Pop": {
    "ID": 78
  },
  "Hip-Hop": {
    "ID": 95
  }
}

URL = 'https://api.radiofrance.fr/livemeta/pull/7'
LOGO_URL = 'https://charte.radiofrance.fr/images/fip/fip.png'

def retrieve():
    data = requests.get(URL).json()
    level = data['levels'][0]
    uid = level['items'][level['position']]
    step = data['steps'][uid]
    return step


def main():
    last_data = None
    notify_id = '0'
    cover_dir = tempfile.mkdtemp() + '/'

    logo = requests.get(LOGO_URL)
    with open(cover_dir + 'fip.jpg', 'wb') as file: 
        file.write(logo.content)
    
    while True:
        try:
            data = retrieve()
        except Exception:
            time.sleep(2)
            continue

        if data != last_data:
            u_data  = {'title': '__titre__', 'authors': '__auteur__', 'anneeEditionMusique': '__année__', 'coverUuid': 'fip.jpg', **data} 
            msg   = "{title} — {authors} ({anneeEditionMusique})".format(**u_data)
            cover =  cover_dir + u_data['coverUuid']

            if (u_data['coverUuid'] != 'fip.jpg'):
                try:
                    img = requests.get(u_data['visual'])
                    with open(cover, 'wb') as file:
                        file.write(img.content)
                except Exception:
                    cover = cover_dir + 'fip.jpg'
            notify_id = subprocess.check_output(['notify-send', '--icon', cover, 
                                                 '--print-id', '--replace-id', notify_id.strip(), 
                                                 'FIP radio',  msg],
                                                  text=True,
                                                )
        last_data = data
        time.sleep(10)


if __name__ == '__main__':
    import traceback
    try:
        main()
    except:
        with open('/tmp/fip-crash.log', 'w') as f:
            traceback.print_exc(file=f)

