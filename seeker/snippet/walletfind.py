#date: 2025-01-01T16:31:30Z
#url: https://api.github.com/gists/06078acea493babe2a7b94311d463c47
#owner: https://api.github.com/users/lyvi

#!/usr/bin/env python3

# Cryptocurrency Wallet Finder
#   Currently supports finding numerous wallet types, such as monero, bitcoin,  ethereum, doge, blockchain, metamask, electrum
#   It will also look for possible mnemonic phrases saved in files
#   
#   Tested on https://github.com/3rdIteration/btcrecover using the test files located in btcrecover/test/test-wallets/ and on a real system.

# Usage examples:
#   python walletfind.py <path> > found.txt
#   python walletfind.py . > found.txt # find in current directory
#   python walletfind.py Data/Old > found.txt # find in Data/Old

# NOTE: THIS WILL PROBABLY TAKE A LONG TIME
#   The script will go through EVERY FILE in the specified directory to look for anything related to cryptocurrency.
#   You can specify file types to ignore in ignore_filetypes and ignore_filenames

# TODO:
#   OCR image indexer for checking through images for mnemonics?

# If this helped you, donations are highly appreciated :D
#   XMR: 82rB82w2UUkN7ytdpV3YpnUNQgXb8hYKjf2Y2a8nKRBh3AtEZuZ4m85TAMdfK2mUaFRwEuH7FhuDF6CBUcJ6uYLg3wHWomg
#   BTC: bc1qamerckatx6v99j40qv29cs9cynrrpzajsgx5u2
#   ETH: 0x863691fAf1d1f5e50E9dA98C3DFbad91eE8eD96D

import os, re, sys, subprocess
from datetime import datetime
start=datetime.now()

if len(sys.argv) < 2:
    sys.exit("File path not specified. Exiting...")

path = sys.argv[1]
files_list = []

popen = subprocess.Popen(["find", path], stdout=subprocess.PIPE)
resultOfSubProcess, errorsOfSubProcess = popen.communicate()
files_list = resultOfSubProcess.decode().splitlines()


ignore_filetypes = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".svg", ".mp3", ".m4a", ".wav", ".ogg", ".mp4", ".webm", ".mov", ".exe", ".dll", ".dmg", ".zip", ".tar", ".gz", ".tgz", ".rar", ".7z", ".html", ".css", ".php", ".js", ".map", ".less", ".py", ".pyc", ".yml", ".tmpl", ".sql", ".conf", ".sh"]
ignore_filenames = ["robots.txt", "thumbs.db", ".DS_Store", "node_modules", "__pycache__", "site-packages", ".npmignore", ".git"]

mnemonic_lenghts = [9, 13, 16, 18, 21, 22, 24, 25]
wallet_files = ["wallet.dat", "aes.json", "userkey.json", "dumpwallet.txt", "wallet-backup", "wallet-backup", "wallet.db", "change.json", "request.json", "legacy.json", "dump.txt", "privkeys.txt", "wallet-data.json", "main_dump.txt", "secondpass_dump.txt", "wallet.android", "wallet.desktop", "wallet-android-backup", "electrum", "identity.json", "wallet.aes", "metamask", "wallet.vault", "bitcoin", "bitcoinj", ".encrypted", "myetherwallet", "mnemonic", ".keys", "kdfparams"]
wallet_strings = ["keymeta!", ".multibit.", "bitcoin", "hd_account", "hdm_address", "pbkdf2_iterations", "bitcoin.main", "addr_history", "master_public_key", "wallet_type", "{\"data\":", "pendingNativeCurrency", "Block_hashes", "mintpool", "mainnet", "testnet", "ethereum", "Ethereum", "mnemonic", "kdfparams", "additional_seeds", "always_keep_local_backup", "BIP39", "BIP-39", "blockchain", "metamask", "electrum"]

for current_file in files_list:
    # print("Checking " + current_file)
    if os.path.splitext(current_file)[1] not in ignore_filetypes and os.path.basename(current_file) not in ignore_filenames:
        # check filenames
        for wallet in wallet_files:
            if wallet in current_file:
                print("[POSSIBLE WALLET] [FILE] " + current_file)

        is_file = os.path.isfile(current_file)
        if is_file:
            num_words = 0
            try:
                current_file_binary = open(current_file, 'rb')
                current_file_binary_data = current_file_binary.read()
            except:
                #print("Could not open " + current_file)
                pass
            
            # check for possible mnemonic
            try:
                words = current_file_binary_data.split()
                num_words += len(words)
                if num_words in mnemonic_lenghts:
                    print("[POSSIBLE MNEMONIC] " + str(num_words) + " WORDS found in " + current_file)
            except:
                #print("Could not check " + current_file)
                pass

            # check file contents
            try:
                for wallet_data in wallet_strings:
                    if re.search(wallet_data, str(current_file_binary_data), re.IGNORECASE):
                        print("[POSSIBLE WALLET] [DATA] \"" + wallet_data + "\" found in " + current_file)
            except:
                #print("Could not check " + current_file)
                pass

            # close the file
            try:
                current_file_binary.close()
            except:
                #print("Could not close " + current_file)
                pass
    # else:
    #     print("Ignoring " + current_file)
print("\nCompleted in " + str(datetime.now()-start))