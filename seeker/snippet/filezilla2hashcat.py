#date: 2024-02-02T16:49:28Z
#url: https://api.github.com/gists/fd650883c1d13c52b09beac64681ec03
#owner: https://api.github.com/users/Yeeb1

import argparse

def main():
    parser = argparse.ArgumentParser(description="Converts FileZilla hashes into a format compatible with Hashcat mode 10900 for cracking. Accepts a hash, a salt, and an optional number of iterations (default is 100000).")
    
    parser.add_argument("salt", help="The salt")
    parser.add_argument("hash", help="The hash value")
    parser.add_argument("--iterations", type=int, default=100000, help="Number of iterations (default: 100000)")

    args = parser.parse_args()

    print(f"sha256:{args.iterations}:{args.salt}:{args.hash}")

if __name__ == "__main__":
    main()
