#date: 2025-05-22T16:39:49Z
#url: https://api.github.com/gists/34ec956a4cf4c438973b79d370ec7837
#owner: https://api.github.com/users/adithya2306

#!/usr/bin/env python3

import re
import sys
import xml.etree.ElementTree as ET
from typing import Dict, List, Union
# pip install cryptography
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend


def clean_key(key_text: str) -> str:
    """Removes PEM headers/footers and all whitespace from a key string."""
    key_text = re.sub(r"-+BEGIN.*?-+|-+END.*?-+", "", key_text, flags=re.DOTALL)
    return "".join(key_text.split())


def convert_pkcs8_to_pkcs1(pem_str: str) -> str:
    """Converts a PKCS#8 PEM private key string to PKCS#1 PEM format."""
    private_key = serialization.load_pem_private_key(
        data=pem_str.encode('utf-8'),
        password= "**********"
        backend=default_backend()
    )
    pkcs1_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    return pkcs1_bytes.decode('utf-8')


def parse_keybox_xml(xml_file: str) -> Dict[str, Dict[str, Union[str, List[str]]]]:
    """Parses keybox xml and returns it as a json dict."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        result = {
            "EC": {"PRIV": "", "CERTS": ["", "", ""]},
            "RSA": {"PRIV": "", "CERTS": ["", "", ""]},
        }
        for key in root.findall(".//Key"):
            algorithm = key.get("algorithm")
            if algorithm in ["ecdsa", "rsa"]:
                key_type = "EC" if algorithm == "ecdsa" else "RSA"
                private_key = convert_pkcs8_to_pkcs1(key.find(".//PrivateKey").text)
                result[key_type]["PRIV"] = clean_key(private_key)
                certs = key.findall(".//Certificate")
                for i, cert in enumerate(certs[:3]):
                    result[key_type]["CERTS"][i] = clean_key(cert.text)
        return result
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML file: {e}")
    except AttributeError as e:
        raise ValueError(f"Unexpected XML structure: {e}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert-keybox.py keybox.xml")
        print("Converts keybox XML file to json")
        return 1

    try:
        # Parse the XML file
        keybox_data = parse_keybox_xml(sys.argv[1])
        print(keybox_data)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    main()