#date: 2022-05-27T17:13:30Z
#url: https://api.github.com/gists/6c982ebb1ede6e5adfc149be15bbde6b
#owner: https://api.github.com/users/simplyequipped

# install packages:
# pip3 install fskmodem tcpkissserver
import fskmodem
import tcpkissserver

# find USB sound card alsa device
alsa_dev = fskmodem.get_alsa_dev('USB PnP')

# start FSK modem at 1200 baud using USB sound card
modem = fskmodem.Modem(alsa_dev=alsa_dev, baudrate=1200)
# start TCP KISS server at 127.0.0.1:8001
server = tcpkissserver.Server(tx_callback=modem.send)
modem.set_rx_callback(server.receive)