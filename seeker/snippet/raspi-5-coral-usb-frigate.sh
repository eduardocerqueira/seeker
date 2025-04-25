#date: 2025-04-25T17:12:48Z
#url: https://api.github.com/gists/095a053a38bd5fa0101366777932dd65
#owner: https://api.github.com/users/Jared-Is-Coding

Command
# = Comment
## = Comment
# -> Something to do
``` File Contents ```

# ==========================================================================================
# BASE
# ==========================================================================================

sudo apt-get update && sudo apt-get upgrade -y

sudo nano /boot/firmware/config.txt
# -> Append after "[all]"
```
usb_max_current_enable=1
dtparam=pciex1_gen=3
kernel=kernel8.img
dtoverlay=pineboards-hat-ai
```

# ==========================================================================================
# DOCKER
# ==========================================================================================

cd ~/Downloads
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh ./get-docker.sh --dry-run
sudo sh ./get-docker.sh

# ==========================================================================================
# EDGE TPU PREREQS
# ==========================================================================================
# Adapted from:
# https://www.diyengineers.com/2024/05/18/setup-coral-tpu-usb-accelerator-on-raspberry-pi-5/
# ==========================================================================================

mkdir -p ~/Docker/hello-world
cd ~/Docker/hello-world
sudo docker run hello-world

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std

sudo reboot

# -> Allow the pi to reboot

# ==========================================================================================
# EDGE TPU
# ==========================================================================================

lsusb
mkdir -p ~/Docker/Deb10
nano ~/Docker/Deb10/Dockerfile

# -> Dockerfile
```
FROM debian:10
  
WORKDIR /home
ENV HOME /home
RUN cd ~
RUN apt-get update
RUN apt-get install -y git nano python3-pip python-dev pkg-config wget usbutils curl
  
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
| tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
RUN apt-get install -y edgetpu-examples
RUN apt-get install -y udev
RUN apt-get install -y sudo
RUN echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="18d1", ATTR{idProduct}=="9302", MODE="0666"' > /etc/udev/rules.d/CORALUSB
```
# -> ctrl + x
# -> y
# -> enter

cd ~/Docker/Deb10
sudo docker build -t "coral" .

# -> Allow the build to finish

# Start the container and enter it
sudo docker run -it --device /dev/bus/usb:/dev/bus/usb coral /bin/bash
# You're executing this command Within the container, now
python3 /usr/share/edgetpu/examples/classify_image.py --model /usr/share/edgetpu/examples/models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite --label /usr/share/edgetpu/examples/models/inat_bird_labels.txt --image /usr/share/edgetpu/examples/images/bird.bmp
## At first, the output will most likely be an error such as this: RuntimeError: Error in device opening (/sys/bus/usb/devices/4-1)!

# -> ctrl + d

lsbusb
# You should now see the desired USB device (Google Inc)

# Start the container and enter it
sudo docker run -it --device /dev/bus/usb:/dev/bus/usb coral /bin/bash
# You're executing this command Within the container, now
$python3 /usr/share/edgetpu/examples/classify_image.py --model /usr/share/edgetpu/examples/models/mobilenet_v2_1.0_224_inat_bird_quant_edgetpu.tflite --label /usr/share/edgetpu/examples/models/inat_bird_labels.txt --image /usr/share/edgetpu/examples/images/bird.bmp
## You should now get the desired output

# ==========================================================================================
# DOCKER COMPOSE
# ==========================================================================================

mkdir -p ~/Docker/frigate
nano ~/Docker/frigate/compose.yaml

# -> compose.yaml
```
version: "3.9"
services:
  frigate:
    container_name: frigate
    privileged: true # this may not be necessary for all setups
    restart: unless-stopped
    stop_grace_period: 30s # allow enough time to shut down the various services
    image: ghcr.io/blakeblackshear/frigate:stable
    shm_size: "512mb" # update for your cameras based on calculation above
    devices:
      - /dev/bus/usb:/dev/bus/usb # Passes the USB Coral, needs to be modified for other versions
      # - /dev/apex_0:/dev/apex_0 # Passes a PCIe Coral, follow driver instructions here https://coral.ai/docs/m2/get-started/#2a-on-linux
      # - /dev/video11:/dev/video11 # For Raspberry Pi 4B
      - /dev/dri/renderD128:/dev/dri/renderD128 # For intel hwaccel, needs to be updated for your hardware
    volumes:
      # - /dev/bus/usb:/dev/bus/usb # Passes the USB Coral, needs to be modified for other versions
      - /etc/localtime:/etc/localtime:ro
      - ./config:/config
      - ./storage:/media/frigate
      - type: tmpfs # Optional: 1GB of memory, reduces SSD/SD Card wear
        target: /tmp/cache
        tmpfs:
          size: 1000000000
    ports:
      - "8971:8971"
      - "5000:5000" # Internal unauthenticated access. Expose carefully.
      - "8554:8554" # RTSP feeds
      - "8555:8555/tcp" # WebRTC over tcp
      - "8555:8555/udp" # WebRTC over udp

    environment:
      FRIGATE_RTSP_PASSWORD: "**********"
```
# -> ctrl + x
# -> y
# -> enter

sudo docker compose up --detach
sudo docker attach frigate

# -> Navigate to https://address:5000/

# ==========================================================================================
# FRIGATE SETTINGS
# ==========================================================================================

```
mqtt:
  enabled: false

detectors:
  coral:
    type: edgetpu
    device: usb

cameras:
  camera_name:
    enabled: true
    ffmpeg:
      inputs:
        - path: rtsp://user:pass@192.168.1.125/ch0
          roles:
            - detect
    detect:
      enabled: false # enable once the coral TPU is set up
      width: 1024
      height: 576
version: 0.15-1
```5-1
```