#date: 2023-01-02T17:03:42Z
#url: https://api.github.com/gists/abf7f7c591600b2e20fd68817fd40a78
#owner: https://api.github.com/users/PeterForth

# Arduino Installation directory
ARDUINO_DIR=~/opt/arduino-1.8.13
ESP32_DIR=~/.arduino15/packages/esp32
BUILD_DIR=/tmp/.build
CACHE_DIR=/tmp/.cache
SOURCE_DIR=~/esp32c3/esp32forth/ESP32forth

# Cleaning
[ -d "${BUILD_DIR}" ] && rm -rf ${BUILD_DIR}
[ -d "${CACHE_DIR}" ] && rm -rf ${CACHE_DIR}

# Compiling
mkdir -p %{BUILD_DIR} ${CACHE_DIR}
${ARDUINO_DIR}/arduino-builder -compile -logger=human \
  -hardware ${ARDUINO_DIR}/hardware \
  -hardware ~/.arduino15/packages \
  -tools ${ARDUINO_DIR}/tools-builder \
  -tools ${ARDUINO_DIR}/hardware/tools/avr \
  -tools ~/.arduino15/packages \
  -libraries ~/Arduino/libraries \
  -fqbn=esp32:esp32:esp32:PSRAM=disabled,PartitionScheme=default,CPUFreq=240,FlashMode=qio,FlashFreq=80,FlashSize=4M,UploadSpeed=921600,LoopCore=1,EventsCore=1,DebugLevel=none \
  -ide-version=10813 \
  -build-path ${BUILD_DIR} \
  -warnings=none \
  -build-cache ${CACHE_DIR} \
  -verbose \
  ${SOURCE_DIR}/ESP32forth.ino
# [...]
# Sketch uses 974737 bytes (74%) of program storage space. Maximum is 1310720 bytes.
# Global variables use 37764 bytes (11%) of dynamic memory, leaving 289916 bytes for local variables. Maximum is 327680 bytes.


# Flashing
python ${ESP32_DIR}/tools/esptool_py/3.1.0/esptool.py \
  --chip esp32 --port /dev/ttyUSB0 --baud 921600 --before default_reset --after hard_reset write_flash \
  -z --flash_mode dio --flash_freq 80m --flash_size detect \
  0xe000 ${ESP32_DIR}/hardware/esp32/2.0.0/tools/partitions/boot_app0.bin \
  0x1000 ${BUILD_DIR}/ESP32forth.ino.bootloader.bin \
  0x10000 ${BUILD_DIR}/ESP32forth.ino.bin 0x8000 ${BUILD_DIR}/ESP32forth.ino.partitions.bin
