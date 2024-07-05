#date: 2024-07-05T16:57:43Z
#url: https://api.github.com/gists/77dccbe8da21f9c90cb35bb6e6b213ed
#owner: https://api.github.com/users/JarrettR

pip3 install -y adafruit-python-shell
git clone https://github.com/adafruit/Raspberry-Pi-Installer-Scripts/
cd Raspberry-Pi-Installer-Scripts/
#https://www.tal.org/tutorials/setup-raspberry-pi-headless-use-usb-serial-console
#cmdline additions:
sed -i 's/fbcon=font:VGA8x8/fbcon=font:VGA8x8 modules-load=dwc2,g_serial/g' adafruit-pitft.py
#boot additions
sed -i 's/dtparam=i2c_arm=on/dtparam=i2c_arm=on\ndtoverlay=dwc2/g' adafruit-pitft.py
systemctl enable getty@ttyGS0.service
python adafruit-pitft.py