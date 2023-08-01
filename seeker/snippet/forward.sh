#date: 2023-08-01T16:52:44Z
#url: https://api.github.com/gists/89f825edb523c18dbda40c87d9ad3ea6
#owner: https://api.github.com/users/MrTuckie

netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=502 connectaddress=192.168.0.35 connectport=502

netsh interface portproxy delete v4tov4 listenaddress=192.168.0.35 listenport=502

netsh interface portproxy show all