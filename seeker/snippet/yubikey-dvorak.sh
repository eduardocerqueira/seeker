#date: 2022-07-22T17:13:31Z
#url: https://api.github.com/gists/30f8cf46a19956943de61074a146d6f9
#owner: https://api.github.com/users/april

# note that this only works with some Yubikeys -- I have confirmed that it works
# fine with my Nano, and it should also work with the NEO, and as is documented
# here by Yubico: https://www.yubico.com/blog/yubikey-keyboard-layouts/

# first, install the ykpersonalize tool:
$ brew install yubikey-personalization

# next, we run the yubikey tool to update its internal keyboard scan map to Dvorak:
$ ykpersonalize -S0c110b071c180d0a0619130f120e09378c918b879c988d8a8699938f928e89b7271e1f202122232425269e2b28

# its output should look like this:

# Firmware version 5.4.3 Touch level 1285 Program sequence 1
#
# A new scanmap will be written.
#
# Commit? (y/n) [n]: y

# note that you can do other layouts as well, such as French AZERTY:
$ ykpersonalize -S06050708090a0b0c0d0e0f111517181986858788898a8b8c8d8e8f9195979899a79e9fa0a1a2a3a4a5a6382b28

# or Turkish QWERTY:
$ ykpersonalize -S06050708090a0b340d0e0f111517181986858788898a8b8c8d8e8f9195979899271e1f202122232425269e2b28