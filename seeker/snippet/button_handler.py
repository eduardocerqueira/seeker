#date: 2022-01-18T17:09:29Z
#url: https://api.github.com/gists/9bbd4c5a8dce4cc401585c0bbe2f31c3
#owner: https://api.github.com/users/jjfalling

#!/usr/bin/env python3
# Toggles rhasspy wake word detection with a button. 
# Intended to work with a v1 google aiy voice hat, but should work with any button and led.

import RPi.GPIO as GPIO
import time
import requests
import signal
import sys
import logging

# ########################################################################################
# Change the following

# gpio pins for button and led
BUTTON_PIN = 23
LED_PIN = 25

# Rhasspy url including protocol and port
RHASSPY_URL = 'http://localhost:12101'

# Specify a site id to apply the wake word state to. Otherwise set to a blank string and will be applied to all sites.
#  https://rhasspy.readthedocs.io/en/latest/reference/#api_listen_for_wake
SITE_ID = 'satellite-livingroom'

# ########################################################################################

TTS_API_URL = RHASSPY_URL + '/api/text-to-speech'
WAKE_API_URL = RHASSPY_URL + '/api/listen-for-wake'
if SITE_ID:
    WAKE_API_URL = WAKE_API_URL + '?siteId=' + SITE_ID

status = "on"


def cleanup():
    # cleanup pin settings to prevent errors
    GPIO.cleanup()


def signal_handler(sig_num, frame):
    print('\nExiting due to signal...\n')
    cleanup()
    sys.exit(0)


def flash_led(led_status):
    # blinks BUTTON_PIN led
    sleep_time = 0.2

    GPIO.output(LED_PIN, False)
    time.sleep(sleep_time)
    GPIO.output(LED_PIN, True)
    time.sleep(sleep_time)
    GPIO.output(LED_PIN, False)
    time.sleep(sleep_time)
    GPIO.output(LED_PIN, True)
    time.sleep(sleep_time)
    GPIO.output(LED_PIN, False)
    time.sleep(sleep_time)
    GPIO.output(LED_PIN, True)
    time.sleep(sleep_time)
    GPIO.output(LED_PIN, False)
    time.sleep(sleep_time)

    if led_status == "on":
        GPIO.output(LED_PIN, False)
    else:
        GPIO.output(LED_PIN, True)


logging.basicConfig(
    # default to info as urllib is noisy on debug
    level=logging.INFO,
    format='[%(asctime)s %(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('ButtonHandler')

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

GPIO.setmode(GPIO.BCM)
# Set pin to be an input pin and set initial value to be pulled low (off)
GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
GPIO.setup(LED_PIN, GPIO.OUT)
# rising detection with debouncing
GPIO.add_event_detect(BUTTON_PIN, GPIO.RISING, bouncetime=500)

# wait for rhasspy to start
count = 0
while True:
    try:
        res = requests.get(RHASSPY_URL)
        if res.status_code == 200:
            break
    except Exception as e:
        logger.info('Rhasspy not running or responding yet. Retrying in 10 sec...: {e}'.format(e=e))

    count += 1
    time.sleep(10)

    if count >= 6:
        # if 60s has past, provide user feedback that something is wrong
        flash_led(status)

# Seems it isn't possible to get the current state, so set to on when starting
requests.post(WAKE_API_URL, data=status)
GPIO.output(LED_PIN, False)
logger.info('Button handler started')

while True:
    if GPIO.event_detected(BUTTON_PIN):
        if status == "off":
            new_status = "on"
            voice_feedback = 'Microphone enabled'
        else:
            new_status = "off"
            voice_feedback = 'Microphone disabled'

        try:
            requests.post(WAKE_API_URL, data=new_status)
            status = new_status

            if status == "on":
                GPIO.output(LED_PIN, False)
            else:
                GPIO.output(LED_PIN, True)

            requests.post(TTS_API_URL, data=voice_feedback)

        except Exception as e:
            logger.error('Cannot connect to rhassspy to change settings: {e}'.format(e=e))
            flash_led(status)

    time.sleep(0.05)

cleanup()