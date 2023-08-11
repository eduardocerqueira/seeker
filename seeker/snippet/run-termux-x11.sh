#date: 2023-08-11T16:56:35Z
#url: https://api.github.com/gists/4cdecc5685ff2b37bb742e160c001483
#owner: https://api.github.com/users/docsolarstone

# This setting was provided by the https://ivonblog.com/en-us/posts/termux-proot-distro-debian/
# my attempt to setup proot distro debian with box86 / box64
pulseaudio --start --load="module-native-protocol-tcp auth-ip-acl=127.0.0.1 auth-anonymous=1" --exit-idle-time=-1

pacmd load-module module-native-protocol-tcp auth-ip-acl=127.0.0.1 auth-anonymous=1



export DISPLAY=:1

#termux-x11 :1 &
TERMUX_X11_DEBUG=1 termux-x11 :1 >>termux-x11-debug.log &


virgl_test_server_android &

proot-distro login debian --user solarbaby --shared-tmp