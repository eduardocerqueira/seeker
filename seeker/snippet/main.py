#date: 2025-05-19T16:44:46Z
#url: https://api.github.com/gists/462fafa0ec92b91569e6d46db35ccd7e
#owner: https://api.github.com/users/newt239

#!/usr/bin/env python3
# https://chatgpt.com/share/682b5fe1-79f8-800f-837d-3617688a5e80
"""
touch_card_daemon.py
- NFC/IC カードの UID を取得
- HTTPS POST で業務 API に送信
"""

import logging
import sys
import time
from binascii import hexlify

import requests
from smartcard.CardMonitoring import CardMonitor, CardObserver
from smartcard.util import toHexString

# ------------------ 設定値 ------------------ #
API_ENDPOINT = "https://api.example.com/v1/card-touch"
API_KEY      = "YOUR_API_KEY"          # .env に隔離推奨
REQ_TIMEOUT  = 5                       # 秒
LOGFILE      = "/var/log/card-touch.log"
# ------------------------------------------- #

logging.basicConfig(
    filename=LOGFILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

class TouchObserver(CardObserver):
    """カード挿抜イベントを監視する Observer 実装"""

    def update(self, observable, (added, removed)):
        for card in added:
            try:
                logging.info("Card detected: %s", card)
                connection = card.createConnection()
                connection.connect()

                # PC/SC ESCAPE コマンド: UID 取得 (ACR122U系)。デバイス依存の場合は適宜変更
                GET_UID = [0xFF, 0xCA, 0x00, 0x00, 0x00]
                data, sw1, sw2 = connection.transmit(GET_UID)
                if sw1 == 0x90 and sw2 == 0x00:
                    uid = hexlify(bytearray(data)).decode().upper()
                    logging.info("UID: %s", uid)
                    _post_to_backend(uid)
                else:
                    logging.error("APDU error sw1=%02X sw2=%02X", sw1, sw2)

            except Exception as e:
                logging.exception("Card processing failed: %s", e)

def _post_to_backend(uid: str) -> None:
    """REST API へ HTTPS POST"""
    payload = {
        "uid": uid,
        "device_id": "pi-station-01",
        "timestamp": int(time.time())
    }
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    try:
        resp = requests.post(API_ENDPOINT, json=payload,
                             headers=headers, timeout=REQ_TIMEOUT)
        resp.raise_for_status()
        logging.info("POST success: %s", resp.status_code)
    except requests.RequestException as e:
        logging.error("POST failed: %s", e)

def main():
    card_monitor = CardMonitor()
    card_observer = TouchObserver()
    card_monitor.addObserver(card_observer)

    logging.info("Card touch daemon started.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Daemon terminated by user.")
    finally:
        card_monitor.deleteObserver(card_observer)

if __name__ == "__main__":
    if sys.version_info < (3, 9):
        sys.exit("Python 3.9 以降が必要です。")
    main()
