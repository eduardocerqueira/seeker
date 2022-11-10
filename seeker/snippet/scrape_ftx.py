#date: 2022-11-10T17:06:40Z
#url: https://api.github.com/gists/e804979c15a924f49ef2849049827d0d
#owner: https://api.github.com/users/mehmetcanbudak

import pandas as pd
import time
import requests
import time
import hmac
from requests import Request
import json

api_key = input("Enter api key:")
api_secret = input("Enter api secret: "**********"

def get(url):
    ts = int(time.time() * 1000)
    request = Request('GET', url)
    prepared = request.prepare()
    signature_payload = f'{ts}{prepared.method}{prepared.path_url}'
    if prepared.body:
        signature_payload += prepared.body
    signature_payload = signature_payload.encode()
    signature = "**********"

    headers = {
        "accept": "application/json",
        "Content-Type": "application/json",
        "FTX-KEY": f"{api_key}",
        "FTX-SIGN": f"{signature}",
        "FTX-TS": f"{ts}",
    }

    return requests.get(url, headers=headers)

def get_start_end(url):
    end_time = int(time.time())
    df = pd.DataFrame()
    while True:
        shape_before = df.shape[0]
        req_url = url + f"?start_time=0&end_time={end_time}"
        df_new = pd.DataFrame(get(req_url).json()["result"])
        df = pd.concat([df, df_new])
        df = df.drop_duplicates()
        shape_after = df.shape[0]
        print(end_time, req_url, df_new.shape[0], shape_after - shape_before)
        if shape_after == shape_before:
            break
        end_time = pd.to_datetime(df["time"]).min().timestamp()
    return df

# account details
account_info = get("https://ftx.com/api/account").json()["result"]
json.dump(account_info, open("account_info.json", "w"))

# balances
balances = get("https://ftx.com/api/wallet/all_balances").json()["result"]
json.dump(balances, open("balances.json", "w"))

# Deposits
deposit_history = get_start_end("https://ftx.com/api/wallet/deposits")
deposit_history.to_csv("deposit_history.csv")

# Withdrawals
withdrawal_history = get_start_end("https://ftx.com/api/wallet/withdrawals")
withdrawal_history.to_csv("withdrawal_history.csv")

# borrow history
borrow_history = get_start_end("https://ftx.com/api/spot_margin/borrow_history")
borrow_history.to_csv("borrow_history.csv")

# lend history
lending_history = get_start_end("https://ftx.com/api/spot_margin/lending_history")
lending_history.to_csv("lending_history.csv")

# referral history
referral_history = get_start_end("https://ftx.com/api/referral_rebate_history")
referral_history.to_csv("referral_history.csv")

# fill history
fills = get_start_end("https://ftx.com/api/fills")
fills.to_csv("fills.csv")

# funding payments
funding_payments = get_start_end("https://ftx.com/api/funding_payments")
funding_payments.to_csv("funding_payments.csv")funding_payments.to_csv("funding_payments.csv")