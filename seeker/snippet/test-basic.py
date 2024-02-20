#date: 2024-02-20T17:01:56Z
#url: https://api.github.com/gists/f4d0ffc8f1f5893ad12e671d2b30e8d2
#owner: https://api.github.com/users/jrosa-digiwest

#!/usr/bin/env python3

from logging import basicConfig, debug, info, warning, error, DEBUG
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from enum import Enum
from subprocess import run, CalledProcessError
from time import sleep as time_sleep

from serial import Serial


class COLOR(str, Enum):
    RESET = "\033[0m"
    DEBUG = "\033[94m"
    SUCCESS = "\033[92m"
    WARNING = "\033[93m"
    ERROR = "\033[91m"
    PROMPT = "\033[96m"
    INFO = "\033[97m"


class InputOutput:
    @staticmethod
    def debug(message: str) -> None:
        """ Logs an info message to the console.

        Args:
            message (str): The info message.
        """
        debug(COLOR.DEBUG + "Debug: " + message + COLOR.RESET)

    @staticmethod
    def success(message: str) -> None:
        """ Logs a success message to the console.

        Args:
            message (str): The success message.
        """
        info(COLOR.SUCCESS + "Success: " + message + COLOR.RESET)

    @staticmethod
    def warning(message: str) -> None:
        """ Logs a warning message to the console.

        Args:
            message (str): The warning message.
        """
        warning(COLOR.WARNING + "Warning: " + message + COLOR.RESET)

    @staticmethod
    def error(message: str, exception: Exception = None) -> None:
        """ Logs an error message to the console.

        Args:
            message (str): The error message.
            exception (Exception, optional): The exception that caused the error. Defaults to None.
        """
        error(COLOR.ERROR + "Error: " + message + COLOR.RESET)
        if exception is not None:
            raise exception
    
    @staticmethod
    def prompt(message: str, accept: list[str] = None) -> str:
        """ Prompts the user for input.

        Args:
            message (str): The prompt message.
            accept (list[str], optional): A list of accepted values. None 
            accepts all inputs. Empty list accept no inputs. Defaults to None.

        Returns:
            str: The user's input.
        """
        while True:
            user_input = input(COLOR.PROMPT + ">> " + message + COLOR.RESET)
            if accept is None:
                return user_input
            elif user_input in accept:
                return user_input
            else:
                InputOutput.warning("Invalid input. Accepted values are " + str(accept) + ". Please try again.")

    @staticmethod
    def print(message: str) -> None:
        """ Prints a message to the console.

        Args:
            message (str): The message.
        """
        print(COLOR.INFO + message + COLOR.RESET)


class Utils:
    @staticmethod
    def issue_command(cmd, capture_output: bool = False, elevated: bool = False) -> str:
        # Create the command.
        cmd = ("sudo " if elevated else "") + cmd
        
        InputOutput.debug("Issuing command: " + cmd)

        # Run the command.
        result = run(cmd, shell=True, check=True, capture_output=capture_output)
        if capture_output:
            return result.stdout.decode("utf-8")


def test_usb() -> None:
    option = InputOutput.prompt("Test USB devices?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return

    for i in range(3):
        InputOutput.prompt(f"Insert USB device {i+1} and press `Enter`.", [""])

        try:
            Utils.issue_command("lsusb -t", capture_output=False, elevated=False)
        except CalledProcessError:
            InputOutput.warning(f"Failed to list USB devices.")
        
        InputOutput.prompt(f"Done, remove the USB device {i+1} and press `Enter`.", [""])

        try:
            Utils.issue_command("lsusb -t", capture_output=False, elevated=False)
        except CalledProcessError:
            InputOutput.warning(f"Failed to list USB devices.")

    InputOutput.success(f"USB test done.")

def test_usb_hub() -> None:
    option = InputOutput.prompt("Test USB hub?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return

    InputOutput.debug("Disabling USB hub...")
    Utils.issue_command("/home/digiwest/usb2514-off.sh", capture_output=False, elevated=True)
    time_sleep(1)

    Utils.issue_command("lsusb -t", capture_output=False, elevated=False)

    InputOutput.debug("Enabling USB hub...")
    Utils.issue_command("/home/digiwest/usb2514-on.sh", capture_output=False, elevated=True)
    time_sleep(5)

    Utils.issue_command("lsusb -t", capture_output=False, elevated=False)

    InputOutput.success("USB hub test done.")

def test_connections_legacy() -> None:
    option = InputOutput.prompt("Test connections?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return
    
    Utils.issue_command("nmcli", capture_output=False, elevated=True)
    
    InputOutput.success(f"Connections test done.")
    
def test_pings_legacy() -> None:
    values = {
        "wlan0": "google.pt",
        "eth0": "192.168.1.11",
        "ppp0": "google.pt"
    }

    option = InputOutput.prompt("Test pings?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return

    InputOutput.prompt("Connect the `eth0` to the host computer with static IPv4 set to `192.168.1.11` and press `Enter`.", [""])

    for dev, endpoint in values.items():
        InputOutput.debug(f"Using interface `{dev}` to ping `{endpoint}`.")
        try:
            Utils.issue_command(f"ping -I {dev} -c 5 {endpoint}", capture_output=False, elevated=True)
        except CalledProcessError:
            InputOutput.error(f"Failed to ping `{endpoint}` using interface `{dev}`.")
            continue

    InputOutput.success("Pings test done.")

def test_connections_simple() -> None:
    connections: dict = {
        "Wired connection 1": {
            "interface": "eth0",
            "address": "192.168.1.11"
        },
        "Digiwest": {
            "interface": "wlan0",
            "address": "google.pt"
        },
        "gsm": {
            "interface": "ppp0",
            "address": "google.pt"
        }
    }

    option = InputOutput.prompt("Test connections?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return

    InputOutput.prompt("Connect the `eth0` to the host computer with static IPv4 set to `192.168.1.11` and press `Enter`.", [""])

    for connection in connections:
        try:
            Utils.issue_command(f"nmcli c down {connection}", capture_output=False, elevated=True)
        except CalledProcessError:
            InputOutput.error(f"Failed to down connection `{connection}`.")
            continue
    
    for connection in connections:
        try:
            Utils.issue_command(f"nmcli c up {connection}")
        except CalledProcessError:
            InputOutput.error(f"Failed to up connection `{connection}`.")
            continue
        
        time_sleep(5)

        try:
            Utils.issue_command(f"ping -I {connections[connection]['interface']} -c 5 {connections[connection]['address']}", capture_output=False, elevated=True)
        except CalledProcessError:
            InputOutput.error(f"Failed to ping `{connections[connection]['address']}` using interface `{connections[connection]['interface']}.")
        
        try:
            Utils.issue_command(f"nmcli c down {connection}", capture_output=False, elevated=True)
        except CalledProcessError:
            InputOutput.error(f"Failed to down connection `{connection}`.")
    
    InputOutput.success(f"Connections test done.")

def test_connections() -> None:
    option = InputOutput.prompt("Test connections?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return

    # Enabled Wifi devices.
    try:
        Utils.issue_command("nmcli radio wifi on", capture_output=False, elevated=True)
    except CalledProcessError:
        InputOutput.error("Failed to enable WIFI connections.")

    # Get all connections.
    try:
        lines = Utils.issue_command("nmcli --terse c", capture_output=True, elevated=True).split("\n")
    except CalledProcessError:
        InputOutput.error("Failed to list connections.")
        return

    connections = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        con_name, con_uuid, con_type = line.split(":", maxsplit=2)

        try:
            inner_lines = Utils.issue_command(f"nmcli --terse --fields=connection.interface-name,ipv4.gateway c show {con_uuid}", capture_output=True, elevated=True).split("\n")
        except CalledProcessError:
            InputOutput.error(f"Failed to get connection `{con_name}` interface and gateway.")
            continue
        
        for inner_line in inner_lines:
            inner_line = inner_line.strip()
            if not inner_line:
                continue

            if inner_line.startswith("connection.interface-name"):
                con_interface = inner_line.split(":", maxsplit=1)[1].strip()
            
            if inner_line.startswith("ipv4.gateway"):
                con_gateway = inner_line.split(":", maxsplit=1)[1].strip()
                
        if not con_gateway:
            con_gateway = "google.pt"
        
        InputOutput.debug(f"Using interface `{con_interface}` to ping `{con_gateway}`.")

        connections.append({
            "name": con_name,
            "uuid": con_uuid,
            "type": con_type,
            "interface": con_interface,
            "gateway": con_gateway
        })

    # Disable all connections.
    for connection in connections:
        try:
            Utils.issue_command(f"nmcli c down {connection['uuid']}", capture_output=False, elevated=True)
        except CalledProcessError:
            InputOutput.error(f"Failed to down connection `{connection['name']}`.")
            continue
    
    # Enable and ping all connections.
    for connection in connections:
        try:
            Utils.issue_command(f"nmcli c up {connection['uuid']}", capture_output=False, elevated=True)
        except CalledProcessError:
            InputOutput.error(f"Failed to up connection `{connection['name']}`.")
            continue

        time_sleep(5)

        try:
            Utils.issue_command(f"ping -I {connection['interface']} -c 5 {connection['gateway']}", capture_output=False, elevated=True)
        except CalledProcessError:
            InputOutput.error(f"Failed to ping `{connection['gateway']}` using interface `{connection['interface']}.")
        
        try:
            Utils.issue_command(f"nmcli c down {connection['uuid']}", capture_output=False, elevated=True)
        except CalledProcessError:
            InputOutput.error(f"Failed to down connection `{connection['name']}`.")

    # Enable all connections.
    for connection in connections:
        try:
            Utils.issue_command(f"nmcli c up {connection['uuid']}", capture_output=False, elevated=True)
        except CalledProcessError:
            InputOutput.error(f"Failed to up connection `{connection['name']}`.")

    # Disable WIFI device?
    option = InputOutput.prompt("Disable all WIFI connections?", ["y", "Y", "n", "N"]).strip()
    if option in ["y", "Y"]:
        try:
            Utils.issue_command("nmcli radio wifi off", capture_output=False, elevated=True)
        except CalledProcessError:
            InputOutput.error("Failed to disable WIFI connections.")

    InputOutput.success(f"Connections test done.")

def test_lte_modem() -> None:
    option = InputOutput.prompt("Test LTE modem?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return
    
    InputOutput.debug("Disabling wireless communications of LTE modem...")
    Utils.issue_command("/home/digiwest/gsm-woff.sh", capture_output=False, elevated=True)
    time_sleep(5)

    Utils.issue_command("nmcli", capture_output=False, elevated=False)

    InputOutput.debug("Disabling LTE modem...")
    Utils.issue_command("/home/digiwest/gsm-off.sh", capture_output=False, elevated=True)
    time_sleep(5)

    Utils.issue_command("lsusb -t", capture_output=False, elevated=False)

    InputOutput.debug("Enabling LTE modem...")
    Utils.issue_command("/home/digiwest/gsm-on.sh", capture_output=False, elevated=True)
    time_sleep(10)

    Utils.issue_command("lsusb -t", capture_output=False, elevated=False)

    InputOutput.debug("Enabling wireless communications of LTE modem...")
    Utils.issue_command("/home/digiwest/gsm-won.sh", capture_output=False, elevated=True)
    time_sleep(10)

    Utils.issue_command("nmcli", capture_output=False, elevated=False)

    InputOutput.success("LTE modem test done.")

def test_lte_modem_simple() -> None:
    option = InputOutput.prompt("Test LTE modem?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return
    
    InputOutput.debug("Disabling wireless communications of LTE modem...")
    Utils.issue_command("/home/digiwest/gsm-woff.sh", capture_output=False, elevated=True)
    time_sleep(2)

    Utils.issue_command("lsusb -t", capture_output=False, elevated=False)

    InputOutput.debug("Enabling wireless communications of LTE modem...")
    Utils.issue_command("/home/digiwest/gsm-won.sh", capture_output=False, elevated=True)
    time_sleep(2)

    Utils.issue_command("lsusb -t", capture_output=False, elevated=False)

    InputOutput.success("LTE modem test done.")

def test_audio() -> None:
    option = InputOutput.prompt("Test audio?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return

    InputOutput.debug("Disabling amplifier and muting audio...")
    Utils.issue_command("/home/digiwest/amp-off.sh", capture_output=False, elevated=True)
    Utils.issue_command("/home/digiwest/amp-mute.sh", capture_output=False, elevated=True)
    time_sleep(1)

    InputOutput.debug("Amplifier disabled and muted, playing audio `test.wav`.")
    Utils.issue_command("aplay /home/digiwest/test.wav", capture_output=False, elevated=True)
    
    InputOutput.debug("Unmuting amplifier...")
    Utils.issue_command("/home/digiwest/amp-unmute.sh", capture_output=False, elevated=True)
    time_sleep(1)

    InputOutput.debug("Amplifier disabled and unmuted, playing audio `test.wav`.")
    Utils.issue_command("aplay /home/digiwest/test.wav", capture_output=False, elevated=True)

    InputOutput.debug("Enabling amplifier...")
    Utils.issue_command("/home/digiwest/amp-on.sh", capture_output=False, elevated=True)
    time_sleep(1)

    InputOutput.debug("Amplifier enabled and unmuted, playing audio `test.wav`.")
    Utils.issue_command("aplay /home/digiwest/test.wav", capture_output=False, elevated=True)

    InputOutput.success("Amplifier test done.")

def test_rtc() -> None:
    option = InputOutput.prompt("Test RTC?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return
    
    rtcs = ["/dev/rtc0", "/dev/rtc1"]

    for rtc in rtcs:
        Utils.issue_command(f"hwclock -f {rtc}", capture_output=False, elevated=True)

    InputOutput.success("RTC test done.")

def test_ntp() -> None:
    option = InputOutput.prompt("Test NTP?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return

    Utils.issue_command("timedatectl | grep --color -i ntp", capture_output=False, elevated=True)
    
    InputOutput.success("NTP test done.")

def test_timedate() -> None:
    option = InputOutput.prompt("Test timedate?", ["y", "Y", "n", "N"]).strip()

    if option in ["n", "N"]:
        return

    Utils.issue_command("timedatectl | grep --color -i time", capture_output=False, elevated=True)
    
    InputOutput.success("Timedate test done.")

def test_gps(port: str = "/dev/ttymxc3", baudrate: int = 9600):
    option = InputOutput.prompt("Test GPS?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return
    
    port: Serial = Serial(port=port, baudrate=baudrate, timeout=5.0, write_timeout=5.0, exclusive=False)
    
    InputOutput.debug("Disabling GPS...")
    Utils.issue_command("/home/digiwest/gps-off.sh", capture_output=False, elevated=True)
    time_sleep(1)

    # Clear the buffer.
    if port.in_waiting:
        port.readall()

    count = 0
    output = ""
    while True:
        if port.in_waiting:
            output = port.readall().decode(errors="ignore").strip()
        
        if output:
            InputOutput.error("Unable to disable GPS: Test failed.")
            break

        if count > 999:
            InputOutput.success("GPS disabled successfully: Test passed.")
            break
        
        count += 1

    InputOutput.debug("Enabling GPS...")
    Utils.issue_command("/home/digiwest/gps-on.sh", capture_output=False, elevated=True)
    time_sleep(1)

    # Clear the buffer.
    if port.in_waiting:
        port.readall()

    count = 0
    while True:
        output = port.readline().decode(errors="ignore").strip()

        if output.startswith("$GNRMC"):
            InputOutput.success("GPS enabled successfully: Test passed.")
            break

        if count > 999:
            InputOutput.error("Unable to enable GPS: Test failed.")
            break

        count += 1

    InputOutput.success("GPS test done.")

def test_leds() -> None:
    option = InputOutput.prompt("Test LEDs?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return

    leds = ["/sys/class/leds/led2", "/sys/class/leds/led3"]

    for led in leds:
        InputOutput.debug(f"Testing LED `{led}`.")
        
        try:
            Utils.issue_command(f"echo none > {led}/trigger", capture_output=False, elevated=True)
        except CalledProcessError:
            InputOutput.warning(f"Failed to set trigger to `none` for LED `{led}`.")

        try:
            Utils.issue_command(f"echo 1 > {led}/brightness", capture_output=False, elevated=True)
        except CalledProcessError:
            InputOutput.warning(f"Failed to set brightness to `1` for LED `{led}`.")
        else:
            time_sleep(2)

        try:
            Utils.issue_command(f"echo 0 > {led}/brightness", capture_output=False, elevated=True)
        except CalledProcessError:
            InputOutput.warning(f"Failed to set brightness to `0` for LED `{led}`.")
    
    InputOutput.success("LEDs test done.")

def test_buttons() -> None:
    option = InputOutput.prompt("Test buttons?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return

    buttons = ["/dev/gpiochip0 1", "/dev/gpiochip0 19"]
    
    for button in buttons:
        InputOutput.debug(f"Testing button `{button}` with settings `--active-low --bias=pull-up`.")
        Utils.issue_command(f"gpiomon --active-low --bias=pull-up {button} &", capture_output=False, elevated=True)
    
    InputOutput.success("Buttons test done.")

def test_pioc_inputs(port: str = "/dev/ttymxc1", baudrate: int = 115200) -> None:
    option = InputOutput.prompt("Test PIOCs inputs?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return

    inputs = [
        0b0001,
        0b0010,
        0b0100,
        0b1000
    ]

    port: Serial = Serial(port=port, baudrate=baudrate, timeout=5.0, write_timeout=5.0, exclusive=False)

    for i, _input in enumerate(inputs):
        InputOutput.debug(f"Assert the input `{i+1}`.")
        count = 0
        while True:
            output = port.readline().decode(encoding="latin-1", errors="ignore").strip()
            
            if output == f"incn {_input:#06x} {_input:#06x}":
                InputOutput.success(f"Input `{i+1}` pressed.")
                break
            
            if count > 999:
                InputOutput.error(f"Input `{i+1}` not pressed.")
                break
            
            count += 1
        
        
        InputOutput.debug(f"Release the input `{i+1}`.")
        count = 0
        while True:
            output = port.readline().decode(encoding="latin-1", errors="ignore").strip()
            
            if output == f"incn 0x0000 {_input:#06x}":
                InputOutput.success(f"Input `{i+1}` released.")
                break
            
            if count > 999:
                InputOutput.error(f"Input `{i+1}` not released.")
                break

            count += 1

    InputOutput.success("PIOCs inputs test done.")

def test_pioc_outputs(port: str = "/dev/ttymxc1", baudrate: int = 115200) -> None:
    option = InputOutput.prompt("Test PIOCs outputs?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return

    outputs = [1, 2]

    port: Serial = Serial(port=port, baudrate=baudrate, timeout=5.0, write_timeout=5.0, exclusive=False)

    for output in outputs:
        InputOutput.debug(f"Testing output `{output}`.")

        InputOutput.debug(f"Disabling the output `{output}`.")
        port.write(f"dout {output} 0\r\n".encode("latin-1"))

        InputOutput.prompt(f"Connect the output `{output}` to a measure device and press `Enter`.", [""])

        InputOutput.debug(f"Enabling the output `{output}`.")
        port.write(f"dout {output} 1\r\n".encode("latin-1"))
        time_sleep(5)

        InputOutput.debug(f"Disabling the output `{output}`.")
        port.write(f"dout {output} 0\r\n".encode("latin-1"))
        time_sleep(5)

    InputOutput.success("PIOCs outputs test done.")

def test_pioc_blacklight(port: str = "/dev/ttymxc1", baudrate: int = 115200) -> None:
    option = InputOutput.prompt("Test PIOCs backlight?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return

    blacklights = ["e13a.bl", "e13b.bl", "e32.bl"]

    port: Serial = Serial(port=port, baudrate=baudrate, timeout=5.0, write_timeout=5.0, exclusive=False)

    for blacklight in blacklights:
        InputOutput.debug(f"Testing blacklight `{blacklight}`.")

        InputOutput.debug(f"Disabling the blacklight `{blacklight}`.")
        port.write(f"{blacklight} 0\r\n".encode("latin-1"))

        InputOutput.prompt(f"Connect the blacklight `{blacklight}` to a measure device and press `Enter`.", [""])

        InputOutput.debug(f"Setting the blacklight `{blacklight}`.")
        for brightness in range(0, 101, 1):
            port.write(f"{blacklight} {brightness}\r\n".encode("latin-1"))
            time_sleep(0.1)

        InputOutput.debug(f"Disabling the blacklight `{blacklight}`.")
        port.write(f"{blacklight} 0\r\n".encode("latin-1"))
    
    InputOutput.success("PIOCs blacklights test done.")

def test_lowpower_modes(port: str = "/dev/ttymxc1", baudrate: int = 115200) -> None:
    option = InputOutput.prompt("Test low-power modes?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return

    modes = ["standby", "freeze"]

    port: Serial = Serial(port=port, baudrate=baudrate, timeout=5.0, write_timeout=5.0, exclusive=False)

    for mode in modes:
        InputOutput.debug(f"Requesting wake up to PIOC within 10 seconds...")
        port.write(f"wakeme 10000\r\n".encode("latin-1"))
        time_sleep(1)

        InputOutput.debug(f"Entering low-power mode `{mode}`.")
        Utils.issue_command(f"echo {mode} > /sys/power/state")

    InputOutput.success("Low-power modes test done.")

def set_hostname() -> None:
    option = InputOutput.prompt("Set hostname?", ["y", "Y", "n", "N"]).strip()
    if option in ["n", "N"]:
        return

    # Gets the wlan0 interface mac address.
    try:
        wlan0_mac = Utils.issue_command("cat /sys/class/net/wlan0/address", capture_output=True, elevated=True).strip()
    except CalledProcessError:
        InputOutput.error("Failed to get `wlan0` mac address. Interface `wlan0` doesn't exists?")
        return
    
    InputOutput.debug(f"Mac address of `wlan0` is `{wlan0_mac}`.")
    
    # Prompts the user for the device serial number.
    serial_number = InputOutput.prompt("Insert the device serial number.").strip().lower()

    # Generates the new hostname.
    hostname = "pipe-" + ((wlan0_mac.replace(":", "")[-6:]).strip().upper()) + "-" + serial_number
    InputOutput.debug(f"Setting hostname to `{hostname}`.")

    # Sets the hostname using hostnamectl.
    try:
        Utils.issue_command(f"hostnamectl set-hostname {hostname}", capture_output=False, elevated=True)
    except CalledProcessError:
        InputOutput.error(f"Failed to set hostname to `{hostname}` using `hostnamectl`.")
        return
    
    # Sets the hostname in `/etc/hosts`.
    try:
        with open("/etc/hosts", "r") as fr:
            lines = fr.readlines()
    except OSError:
        InputOutput.error("Failed to read from `/etc/hosts`.")
        return
    
    output_lines: list = [
        "127.0.0.1\tlocalhost\n"
    ]
    for line in lines:
        line = line.strip()
        
        if not line:
            output_lines.append(f"{line}\n")
            continue

        if line.startswith("#"):
            output_lines.append(f"{line}\n")
            continue
        
        config = line.split("#", maxsplit=1)[0]
        config = config.strip()

        ip = config.split(" ", maxsplit=1)[0]
        ip = ip.strip()

        if "127.0." not in ip:
            output_lines.append(f"{line}\n")
            # InputOutput.debug(f"Added `{line}` to `/etc/hosts`.")
        else:
            InputOutput.warning(f"Skipped `{line}` in `/etc/hosts`.")
        
    output_lines.append(f"127.0.1.1\t{hostname}\n")

    # Changes or adds the hostname to an `127.0.1.1` entry in `/etc/hosts`
    try:
        with open("/etc/hosts", "w") as fr:
            fr.writelines(output_lines)
    except OSError:
        InputOutput.error("Failed to write to `/etc/hosts`.")
        return
    
    InputOutput.success(f"Hostname set to `{hostname}`.")

def set_journalctl_size() -> None:
    logger.warning(">> Setting journalctl size.")
    
    Utils.issue_command("sed -i \"s/#SystemMaxUse=/SystemMaxUse=10M/g\" /etc/systemd/journald.conf")
    Utils.issue_command("sudo journalctl --vacuum-size=10M")

    logger.warning("Journalctl size set. Press `Enter` to continue...")
    input()

def set_systemd_log_rotate() -> None:
    logger.warning(">> Setting systemd log rotate.")
    logger.error("Copy the files from `etc-logrotate.d` to `/etc/logrotate.d/`.")
    logger.warning("Systemd log rotate set. Press `Enter` to continue...")
    input()

def clean_var_log() -> None:
    logger.warning(">> Cleaning `/var/log`.")
    Utils.issue_command("sudo rm -f /var/log/*.gz")
    logger.warning("Cleaning done. Press `Enter` to continue...")
    input()

if __name__ == "__main__":
    parser = ArgumentParser(description="Digiwest test script for EPAPER-IMX6 / EPAPER-X2 / DGWCB Board.", formatter_class=ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    
    # Set the InputOutput level.
    basicConfig(level=DEBUG, format="%(levelname)s: %(message)s")

    test_usb()
    test_usb_hub()
    test_connections()
    test_lte_modem_simple()
    test_audio()
    test_rtc()
    test_ntp()
    test_timedate()
    test_gps()
    test_leds()
    test_buttons()
    test_pioc_inputs()
    test_pioc_outputs()
    test_pioc_blacklight()
    # set_journalctl_size()
    # set_systemd_log_rotate()
    # clean_var_log()
    test_lowpower_modes()
    set_hostname()
    
    InputOutput.success("Test done. Issue the command `cat /dev/null > ~/.bash_history && history -c && exit` to clear the bash history and reboot!")
