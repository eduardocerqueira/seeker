#date: 2025-07-30T17:00:29Z
#url: https://api.github.com/gists/12ccac3cde10edf2f37d39406bb96113
#owner: https://api.github.com/users/zyn3rgy

#!/usr/bin/env python3
"""
Netscape to Chrome Cookie Converter

Written to convert cookies in the netscape format to a Chrome-compatible
format with the intention of mass importing the output into Chrome using:

    https://github.com/fkasler/cuddlephish/tree/main/stealerjs_extension



Usage: python netscape2chrome.py input_file output_file
"""

import json
import sys
import argparse
from datetime import datetime
from urllib.parse import urlparse


def parse_netscape_cookies(input_file):
    """
    Parse Netscape format cookies from input file.
    
    Netscape format: domain\tFLAG\tpath\tsecure_connection\texpiration\tname\tvalue
    """
    cookies = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Split by tabs
            parts = line.split('\t')
            if len(parts) != 7:
                print(f"Warning: Line {line_num} has incorrect format, skipping: {line}")
                continue
            
            domain, flag, path, secure, expires, name, value = parts
            
            # Convert Netscape format to Chrome format
            cookie = {
                "domain": domain,
                "expires": 1767243600,  # Hardcoded as specified
                "httpOnly": False,      # Always false as specified
                "name": name,
                "path": path,
                "priority": "Medium",   # Always Medium as specified
                "sameParty": False,     # Always false as specified
                "sameSite": "Lax",      # Default value
                "secure": False,        # Always false as specified
                "session": False,       # Not a session cookie since we have expiration
                "size": len(name) + len(value),  # Calculate size as name + value length
                "sourcePort": 443,      # Always 443 as specified
                "sourceScheme": "Secure",  # Always Secure as specified
                "value": value
            }
            
            cookies.append(cookie)
    
    return cookies


def create_chrome_format(cookies, base_url=None):
    """
    Create Chrome format JSON structure.
    """
    if not base_url:
        # Try to extract base URL from first cookie domain
        if cookies:
            first_domain = cookies[0]["domain"]
            if first_domain.startswith('.'):
                first_domain = first_domain[1:]
            base_url = f"https://{first_domain}/"
        else:
            base_url = "https://example.com/"
    
    chrome_format = {
        "url": base_url,
        "cookies": cookies,
        "localStorage": {}
    }
    
    return chrome_format


def main():
    parser = argparse.ArgumentParser(
        description="Convert Netscape format cookies to Chrome format JSON"
    )
    parser.add_argument("input_file", help="Input file containing Netscape format cookies")
    parser.add_argument("output_file", help="Output file for Chrome format JSON")
    parser.add_argument(
        "--url", 
        help="Base URL for the cookies (optional, will be inferred from domain if not provided)"
    )
    
    args = parser.parse_args()
    
    try:
        # Parse cookies from input file
        cookies = parse_netscape_cookies(args.input_file)
        
        if not cookies:
            print("No valid cookies found in input file")
            sys.exit(1)
        
        # Create Chrome format
        chrome_format = create_chrome_format(cookies, args.url)
        
        # Write to output file
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(chrome_format, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully converted {len(cookies)} cookies to Chrome format")
        print(f"Output written to: {args.output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 