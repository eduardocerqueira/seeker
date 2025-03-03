#date: 2025-03-03T16:54:46Z
#url: https://api.github.com/gists/3162f3f59a0143a0474a217800056332
#owner: https://api.github.com/users/gingerbreadtrev

#!/usr/bin/env python3
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
import binascii
import argparse

def decrypt_shellcode(encrypted_file, key_hex, iv):
    # Convert key and IV from hex to bytes
    # Clean the hex string (remove any whitespace, 0x prefixes, etc.)
    key_hex = key_hex.replace(' ', '').replace('0x', '').replace('\n', '').strip()
    try:
        key = binascii.unhexlify(key_hex)
    except binascii.Error as e:
        raise ValueError(f"Invalid key format: {str(e)}. Key must be valid hexadecimal.")
    
    # Read encrypted data
    with open(encrypted_file, 'rb') as f:
        encrypted_data = f.read()
    
    # Create appropriate cipher based on mode
    if not iv_hex:
        raise ValueError("IV is required for CBC mode")
    # Clean the IV hex string
    iv_hex = iv_hex.replace(' ', '').replace('0x', '').replace('\n', '').strip()
    try:
        iv = binascii.unhexlify(iv_hex)
    except binascii.Error as e:
        raise ValueError(f"Invalid IV format: {str(e)}. IV must be valid hexadecimal.")
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv))

    # Decrypt the data
    try:
        decryptor = cipher.decryptor()
        decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
        
        # Handle potential padding (PKCS7)
        # Since cryptography doesn't expose direct padding removal, we'll use a simple
        # approach to detect and remove PKCS#7 padding if present
        if decrypted_data:
            padding_value = decrypted_data[-1]
            if 1 <= padding_value <= 16:  # AES block size is 16 bytes
                # Check if the last 'padding_value' bytes are all equal to 'padding_value'
                if all(b == padding_value for b in decrypted_data[-padding_value:]):
                    decrypted_data = decrypted_data[:-padding_value]
        
        return decrypted_data
    except Exception as e:
        raise RuntimeError(f"Decryption failed: {str(e)}")

def save_for_analysis(decrypted_data):
    # Save raw binary
    output_file = 'shellcode'
    with open(f"{output_file}.bin", 'wb') as f:
        # Remove dashes and convert to binary
        hex_str = decrypted_data.replace('-', '')
        binary_data = bytes.fromhex(hex_str)
        f.write(binary_data)

    print(f"Raw binary saved to {output_file}.bin")

def main():
    parser = argparse.ArgumentParser(description="Decrypt AES-encrypted shell code")
    parser.add_argument("encrypted_file", help="Path to the encrypted shell code file")
    parser.add_argument("key", help="AES key in hexadecimal format")
    parser.add_argument("iv", help="Initialization vector in hexadecimal format")
    
    args = parser.parse_args()
    
    try:
        # Decrypt the shell code
        decrypted_data = decrypt_shellcode(args.encrypted_file, args.key, args.iv)
        
        # Save the decrypted shell code for analysis
        save_for_analysis(decrypted_data)
        
        print(f"Successfully decrypted {len(decrypted_data)} bytes of shell code")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()