#date: 2025-07-25T17:01:17Z
#url: https://api.github.com/gists/18e1436f34f1468ffea01eef5dd158a0
#owner: https://api.github.com/users/amd-vivekag

import argparse
from transformers import AutoTokenizer

def main():
    # Parse arguments
    parser = "**********"="Decode token IDs into text.")
    parser.add_argument("--model", required=True, help="Model name (e.g. gpt2, meta-llama/Llama-2-7b-hf)", default="meta-llama/Llama-3.1-8B")
    parser.add_argument("--token_file", required= "**********"="Path to file containing token IDs e.g. \"128000, 128000, 128000, 128006, 9125, 128007, 198, 2675, 527, 264, 7701, 27877, 15592, 29385, 2065\"")
    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer for model: "**********"
    tokenizer = "**********"

    # Read token IDs from file
 "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"a "**********"r "**********"g "**********"s "**********". "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"f "**********"i "**********"l "**********"e "**********", "**********"  "**********"" "**********"r "**********"" "**********", "**********"  "**********"e "**********"n "**********"c "**********"o "**********"d "**********"i "**********"n "**********"g "**********"= "**********"" "**********"u "**********"t "**********"f "**********"- "**********"8 "**********"" "**********") "**********"  "**********"a "**********"s "**********"  "**********"f "**********": "**********"
        token_str = "**********"

    # Parse token IDs (supports space/comma-separated)
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"" "**********", "**********"" "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"_ "**********"s "**********"t "**********"r "**********": "**********"
        token_ids = "**********"
    else:
        token_ids = "**********"

    print(f"Token IDs: "**********"

    # Decode tokens into text
    decoded_text = "**********"=True)
    print("\n=== Decoded Text ===")
    print(decoded_text)

if __name__ == "__main__":
    main()

