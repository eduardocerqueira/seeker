#date: 2024-06-26T17:04:34Z
#url: https://api.github.com/gists/ce08f4552be84ee019e1a864baf02b40
#owner: https://api.github.com/users/ErhardRainer

import argparse

def main():
    parser = argparse.ArgumentParser(description='Ein einfaches Skript zum Empfangen benannter Parameter.')
    
    # Definieren der erwarteten Parameter
    parser.add_argument('--Value1', type=str, required=True, help='Der erste Wert')
    parser.add_argument('--Value2', type=str, required=True, help='Der zweite Wert')
    
    # Parsing der Argumente
    args = parser.parse_args()
    
    # Zugriff auf die Parameter
    value1 = args.Value1
    value2 = args.Value2
    
    print(f"Value1: {value1}")
    print(f"Value2: {value2}")

if __name__ == "__main__":
    main()
