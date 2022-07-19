#date: 2022-07-19T16:57:57Z
#url: https://api.github.com/gists/bea4718d05b2502933c7294ef1de16e9
#owner: https://api.github.com/users/vadimstasiev

#!/bin/sh

# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Author: Hexengraf

# Offset values. To search for stable values, use a separate Xorg session with PRIME.
# DO _NOT_ USE UNTESTED VALUES HERE!
glx_offset=170
mem_offset=300
# Maximum performance level (offsets are only assigned to the max level).
# Check the correct value in nvidia-settings, PowerMizer page.
perf_level=4
# Path to nvidia-settings.
# xinit requires a command starting with /, so a full path is necessary.
nvset_path=$(which nvidia-settings)

echo "Configuring Xorg to use PRIME..."
sudo ln -sf /etc/X11/xorg.conf.d/prime.layout /etc/X11/xorg.conf.d/00-layout.conf

# You can add more commands related to overclock here. For instance, cpu undervolting:
# sudo intel-undervolt apply

# First Xorg session: only assigns the offsets and closes.
# You can add more assignments beyond clock offsets. Just append them to the command.
xinit ${nvset_path} -a GPUGraphicsClockOffset[${perf_level}]=${glx_offset} \
                    -a GPUMemoryTransferRateOffset[${perf_level}]=${mem_offset}

echo "Configuring Xorg to use Render Offload..."
sudo ln -sf /etc/X11/xorg.conf.d/offload.layout /etc/X11/xorg.conf.d/00-layout.conf

# Second Xorg session: real session with window manager.
startx
