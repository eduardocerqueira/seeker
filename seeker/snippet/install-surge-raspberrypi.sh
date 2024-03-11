#date: 2024-03-11T17:02:06Z
#url: https://api.github.com/gists/072ee53a29d64f6ae086b38ed13125e1
#owner: https://api.github.com/users/thomasvolk

# install Raspberry Pi OS (64-bit)
# Released 2023-12-05
# use raspberry pi imager https://www.raspberrypi.com/software/

sudo apt install  cmake
sudo apt install build-essential libcairo-dev libxkbcommon-x11-dev libxkbcommon-dev libxcb-cursor-dev libxcb-keysyms1-dev libxcb-util-dev libxrandr-dev libxinerama-dev libxcursor-dev libasound2-dev libjack-jackd2-dev

# checkout surce (commit hash 73bd2a772e6954ff79950ca627dc348150c6f43b)
git clone https://github.com/surge-synthesizer/surge.git
cd surge/
git submodule update --init --recursive
cmake -Bbuild
cmake --build build --config Release --target surge-staged-assets

# cd build/surge_xt_products/
# ./Surge\ XT