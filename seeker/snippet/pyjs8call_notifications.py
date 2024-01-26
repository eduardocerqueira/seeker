#date: 2024-01-26T16:42:26Z
#url: https://api.github.com/gists/fd813e2b4d8748f427e701f0f5a4c217
#owner: https://api.github.com/users/simplyequipped

# 1. Use of notifications requires pyjs8call v0.2.2 or higher
# 2. Remember to configure JS8Call radio and audio settings before using pyjs8call
# 3. Update SMTP email address and password (line 25)
# 4. Update destination phone number and mobile carrier domain, or use an email address (line 27)
# 5. Optionally, uncomment and update station spot configuration (lines 32-34)
#
# Common North America SMS carrier domains:
# Verizon   xxxxxxxxxx@vtext.com (limited to 160 characters)
# AT&T      xxxxxxxxxx@text.att.net
# T-Mobile  xxxxxxxxxx@tmomail.net
# Bell      xxxxxxxxxx@txt.bellmobility.com
# Rogers    xxxxxxxxxx@pcs.rogers.com
#
# See https://simplyequipped.github.io/pyjs8call/ for official pyjs8call docs

import pyjs8call

js8call = pyjs8call.Client()
js8call.start()
# ensure radio is set to the JS8Call freq (helpful for QDX and QMX radios that default to the FT8 freq)
js8call.settings.set_freq(7078000)

# configure and enable incoming directed message notifications
# see pyjs8call.notifications docs for information about Google app passwords
js8call.notifications.set_smtp_credentials('email.address@gmail.com', 'app_password')
# set destination phone number and mobile carrier domain (or email address)
js8call.notifications.set_email_destination('xxxxxxxxxx@vtext.com')
# enable automatic notifications
js8call.notifications.enable()

# configure and enable station spot notifications
#js8call.spots.add_station_watch('OH8STN')
#js8call.spots.add_station_watch('KT7RUN')
#js8call.notifications.station_spots_enabled = True

print('JS8Call is running via pyjs8call with notifications enabled')
input('Press any key to exit...')
# stop the JS8Call application and pyjs8call processes before exiting
js8call.stop()