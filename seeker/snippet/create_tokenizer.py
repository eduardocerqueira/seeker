#date: 2023-12-28T17:08:13Z
#url: https://api.github.com/gists/ffc4b9451669cc52d38b1a45b6835dc9
#owner: https://api.github.com/users/jondurbin

import re
import gc
import os
import glob
import json
from copy import deepcopy
from datasets import concatenate_datasets, Dataset
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# Load a sampling of AR data.
dataset = (
    concatenate_datasets(
        [
            Dataset.from_json(path)
            for path in glob.glob("madlad-ar-sampled/*.jsonl")
            if os.stat(path).st_size
        ]
    )
    .shuffle(seed=42)
    .select(range(1000000))
)

# Yield dataset in batches.
batch_size = 1000


def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["text"]
        gc.collect()


# Download and initialize the original mistral tokenizer.
snapshot_download(
    repo_id="mistralai/Mistral-7B-v0.1",
    local_dir="mistral-7b-v0.1",
    allow_patterns= "**********"
)
tokenizer = "**********"=True)

# The train_new_from_iterator method ignores previous tokens, so the vocab_size param should
# be the number of new tokens desired, not total tokens, in this case 2*16.  You can limit
# the maximum size of new tokens with max_token_length as well, which helps prevent commonly
# repeated URLs, disclaimers, and other noise from entering the vocab.
new_vocab_size = "**********"
new_tokenizer = "**********"
    batch_iterator(), vocab_size= "**********"=24
)
new_tokenizer.save_pretrained("mistral-7b-tokenizer-ar-temp")

# Load the original tokenizer.
 "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"" "**********"m "**********"i "**********"s "**********"t "**********"r "**********"a "**********"l "**********"- "**********"7 "**********"b "**********"/ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********". "**********"j "**********"s "**********"o "**********"n "**********"" "**********") "**********"  "**********"a "**********"s "**********"  "**********"f "**********": "**********"
    original = json.load(f)

# Load the updated tokenizer we just trained.
 "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"" "**********"m "**********"i "**********"s "**********"t "**********"r "**********"a "**********"l "**********"- "**********"7 "**********"b "**********"- "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********"- "**********"a "**********"r "**********"- "**********"t "**********"e "**********"m "**********"p "**********"/ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********". "**********"j "**********"s "**********"o "**********"n "**********"" "**********") "**********"  "**********"a "**********"s "**********"  "**********"f "**********": "**********"
    append = json.load(f)


def merge_tokenizer(original_data: "**********": dict):
    original_vocab = original_data["model"]["vocab"]
    append_vocab = append_data["model"]["vocab"]

    vocab_out = deepcopy(original_vocab)
    data_out = deepcopy(original_data)

    idx = max(vocab_out.values())

    # Append the new vocab tokens, ignoring numeric values since they decrease math/reasoning performance.
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"  "**********"i "**********"n "**********"  "**********"a "**********"p "**********"p "**********"e "**********"n "**********"d "**********"_ "**********"v "**********"o "**********"c "**********"a "**********"b "**********". "**********"k "**********"e "**********"y "**********"s "**********"( "**********") "**********": "**********"
        if token not in original_vocab and not (
            re.search(r"[0-9]", token) and re.match(r"^([^\w]|[0-9])+$", token)
        ):
            idx += 1
            vocab_out[token] = "**********"

    # Update merges.
    merges_out = []
    for candidate, piece_id in vocab_out.items():
        for i in range(1, len(candidate)):
            left, right = candidate[:i], candidate[i:]

            left_id = vocab_out.get(left, None)
            right_id = vocab_out.get(right, None)

            if left_id is not None and right_id is not None:
                if (
                    re.search(r"[0-9]", left)
                    and re.match(r"^([^\w]|[0-9])+$", left)
                    and re.search(r"[0-9]", right)
                    and re.match(r"^([^\w]|[0-9])+$", right)
                ):
                    continue
                merges_out += [f"{left} {right}"]

    data_out["model"]["vocab"] = vocab_out
    data_out["model"]["merges"] = merges_out

    tokenizer.save_pretrained("mistral-7b-tokenizer-ar")
 "**********"  "**********"  "**********"  "**********"  "**********"w "**********"i "**********"t "**********"h "**********"  "**********"o "**********"p "**********"e "**********"n "**********"( "**********"" "**********"m "**********"i "**********"s "**********"t "**********"r "**********"a "**********"l "**********"- "**********"7 "**********"b "**********"- "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********"- "**********"a "**********"r "**********"/ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"i "**********"z "**********"e "**********"r "**********". "**********"j "**********"s "**********"o "**********"n "**********"" "**********", "**********"  "**********"" "**********"w "**********"" "**********") "**********"  "**********"a "**********"s "**********"  "**********"f "**********": "**********"
        json.dump(data_out, f, ensure_ascii=False, indent=2)


merge_tokenizer(original_data= "**********"=append)