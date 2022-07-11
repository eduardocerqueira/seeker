#date: 2022-07-11T16:49:31Z
#url: https://api.github.com/gists/63696a40869d24bdfa81c104d52604a8
#owner: https://api.github.com/users/kitek83

# 2_leds_blinking.py
'''
In this program 5 leds connected to respberry pi 4b wil be blinking.
'''
import RPi.GPIO as GPIO
import time


def main():

    # set snetinel
    yes = 'y'

    # infinite loop
    while yes.lower() == 'y':

        if yes.lower() == 'y':

            try:
                tim_e = float(
                    input('Enter ner time in seconds (format: 0.25): '))
                # activate function
                blink(tim_e)
            except ValueError:
                pass

        print()

        # set sentinel
        yes = input(
            "Enter 'y' if you want to enter new time for blinking or press 'n' to finish: ")

    print()
    prinst('Program is finished.')


def blink(tim_e):

    # GPIO's PINS
    LED_PIN1 = 17
    LED_PIN2 = 5
    LED_PIN3 = 26
    LED_PIN4 = 18
    LED_PIN5 = 23

    # stup() and setmode() must be runne only once time not ti give an error
    # refer to the GPIO connections
    GPIO.setmode(GPIO.BCM)

    # setup out pins only one time !
    GPIO.setup((LED_PIN1, LED_PIN2, LED_PIN3, LED_PIN4, LED_PIN5), GPIO.OUT)
    print('Enter only control+c to finish properly the program.')
    print()

    try:
        while True:
            # leds on
            GPIO.output((LED_PIN1, LED_PIN2, LED_PIN3,
                        LED_PIN4, LED_PIN5), GPIO.HIGH)
            time.sleep(tim_e)

            # leds off
            GPIO.output((LED_PIN1, LED_PIN2, LED_PIN3,
                        LED_PIN4, LED_PIN5), GPIO.LOW)
            time.sleep(tim_e)

    # control + c - to rise excpetion
    except KeyboardInterrupt:
        # clear previus gpio's pins settings to be aple to start program again
        GPIO.cleanup()


main()
