#date: 2024-03-26T17:07:26Z
#url: https://api.github.com/gists/7e80c54be568e7c56524255ae44ebb82
#owner: https://api.github.com/users/johnkanagawa

import mnemonic
import bip32utils
import requests
import logging
import time
import colorama

def generate_mnemonic():
    mnemo = mnemonic.Mnemonic("english")
    return mnemo.generate(strength=128)

def recover_wallet_from_mnemonic(mnemonic_phrase):
    seed = mnemonic.Mnemonic.to_seed(mnemonic_phrase)
    root_key = bip32utils.BIP32Key.fromEntropy(seed)
    child_key = root_key.ChildKey(44 | bip32utils.BIP32_HARDEN).ChildKey(0 | bip32utils.BIP32_HARDEN).ChildKey(0 | bip32utils.BIP32_HARDEN).ChildKey(0).ChildKey(0)
    address = child_key.Address()
    balance = check_BTC_balance(address)
    return mnemonic_phrase, balance, address
def check_BTC_balance(address, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = requests.get(f"https://blockchain.info/balance?active={address}", timeout=10)
            response.raise_for_status()
            data = response.json()
            balance = data[address]["final_balance"]
            return balance / 100000000
        except requests.RequestException as e:
            if attempt < retries - 1:
                print(colorama.Fore.YELLOW+f"Error checking balance, retrying in {delay} seconds: {str(e)}")
                time.sleep(delay)
            else:
                print(colorama.Fore.YELLOW+"Error checking balance: %s", str(e))
    return 0

if __name__ == "__main__":
    mnemonic_count = 0
    print(colorama.Fore.LIGHTCYAN_EX+'-'*10+"BTC Wallet Finder"+"-"*10)
    print(colorama.Fore.LIGHTCYAN_EX+"[*] Programme original : https://github.com/MinightDev/BTC-Wallet-Recover")
    while True:
        mnemonic_phrase = generate_mnemonic()
        mnemonic_phrase, balance, address = recover_wallet_from_mnemonic(mnemonic_phrase)
        if balance > 0:
            print(colorama.Fore.RED+"[+] {} : balance : {} | {} : {}".format(mnemonic_count,balance,mnemonic_phrase,address))

            with open("wallet.txt", "a") as f:
                f.write("[+] {} : balance : {} | {} : {}".format(mnemonic_count,balance,mnemonic_phrase,address))
        else:
            print(colorama.Fore.BLUE+"[+] {} : balance : {} | {} : {}".format(mnemonic_count,balance,mnemonic_phrase,address))
        mnemonic_count += 1
