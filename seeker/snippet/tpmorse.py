#date: 2025-01-01T17:04:54Z
#url: https://api.github.com/gists/a9504560f462244fa386ae4f4e42b060
#owner: https://api.github.com/users/theodric

#!/usr/bin/env python3
import sys
import time

# path to the led brightness file
LED_BRIGHTNESS_PATH = "/sys/devices/platform/thinkpad_acpi/leds/tpacpi::lid_logo_dot/brightness"

# morse code dictionary
MORSE_CODE = {
    'a': ".-", 'b': "-...", 'c': "-.-.", 'd': "-..", 'e': ".", 'f': "..-.",
    'g': "--.", 'h': "....", 'i': "..", 'j': ".---", 'k': "-.-", 'l': ".-..",
    'm': "--", 'n': "-.", 'o': "---", 'p': ".--.", 'q': "--.-", 'r': ".-.",
    's': "...", 't': "-", 'u': "..-", 'v': "...-", 'w': ".--", 'x': "-..-",
    'y': "-.--", 'z': "--..", '1': ".----", '2': "..---", '3': "...--",
    '4': "....-", '5': ".....", '6': "-....", '7': "--...", '8': "---..",
    '9': "----.", '0': "-----", ' ': " "
}

def get_led_state():
    # Retrieve the current LED state so we can tidy it up after the program exits, and bark at the user if our power level is too low
    try:
        with open(LED_BRIGHTNESS_PATH, 'r') as led:
            return led.read().strip()
    
    except PermissionError:
        print("\n\nA C C E S S   D E N I E D\n\n\nEither run the program as root, or set perms to o+w on /sys/devices/platform/thinkpad_acpi/leds/tpacpi::lid_logo_dot/brightness\n\n")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def set_led(state):
    # Check and see if we can even control the LED, and bark at the user if our power level is too low
    try:
        with open(LED_BRIGHTNESS_PATH, 'w') as led:
            led.write("1" if state else "0")
    except PermissionError:
        print("\n\nA C C E S S   D E N I E D\n\n\nEither run the program as root, or set perms to o+w on /sys/devices/platform/thinkpad_acpi/leds/tpacpi::lid_logo_dot/brightness\n\n")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def blink_morse(message):
    # Commence Morsification
    print('\nTransmitting message: "' + message.lower() + '"')

    # Define a fixed cell width for each letter pair printed, enough to accommodate the widest Morse letter 0 == '-----'
    CELL_WIDTH = 5  

    latin_row = []
    morse_row = []

    for char in message.lower():
        if char not in MORSE_CODE:
            continue

        code = MORSE_CODE[char]
        latin_row.append(f"{char.upper():<{CELL_WIDTH}}")
        morse_row.append(f"{code:<{CELL_WIDTH}}")

        print("".join(latin_row))
        print("".join(morse_row))

        for symbol in code:
            time.sleep(0.4)  # intra-character gap
            if symbol == ".":
                set_led(1)
                time.sleep(0.3)
                set_led(0)
            elif symbol == "-":
                set_led(1)
                time.sleep(0.9)
                set_led(0)

        time.sleep(0.5)  # inter-character gap

    # clear the last two rows for the next update, but leave the final message onscreen at exit.
#    if char != message.lower().strip()[-1]:
        print("\033[F\033[K", end="")  # move cursor up and clear line
        print("\033[F\033[K", end="")  # move cursor up and clear line

if __name__ == "__main__":
    # Save the current state of the LED for later use
    initial_state = get_led_state()
    set_led(0)  # switch LED off for second before starting, just in case
    time.sleep(1)
    if len(sys.argv) < 2:
        print("Usage: tpmorse.py <message>")
        sys.exit(1)

    message = " ".join(sys.argv[1:])
    blink_morse(message)
    print("Over.\nResetting LED to previous state after 2 seconds so as not to confuse the operator on the receiving end.")
    time.sleep(2)
    set_led(int(initial_state))
    print("Done, 73\n")
