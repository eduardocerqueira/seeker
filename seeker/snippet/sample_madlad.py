#date: 2023-12-28T17:08:13Z
#url: https://api.github.com/gists/ffc4b9451669cc52d38b1a45b6835dc9
#owner: https://api.github.com/users/jondurbin

import glob
import subprocess
import datasets
import os
from data_selection import HashedNgramDSIR
from huggingface_hub import snapshot_download
from loguru import logger

# Download the AR data, as well as a small sampling of EN data.
logger.info("Downloading data files...")
snapshot_download(
    repo_id="allenai/madlad-400",
    local_dir="madlad-400",
    cache_dir=".cache",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "data/ar/ar_clean_*.gz",
        "data/en/en_clean_000*.gz",
    ],
    repo_type="dataset",
)

logger.info("Extracting gzips...")
current_path = os.getcwd()
for path in map(str, glob.glob("madlad-400/data/*/*clean*.gz", recursive=True)):
    logger.info(f"Extracting: {path}")
    os.chdir(os.path.dirname(os.path.abspath(path)))
    subprocess.run(["gunzip", path])

# Sample AR datasets.
logger.info("Sampling AR datasets...")

# Need to initialize an empty target dataset file - if you want to extract
# data with a relative "importance", you can populate this with the target data.
with open("madlad-400-ar-sample.jsonl", "a+") as outfile:
    ...

# Filter down the AR dataset via DSIR to N documents.
ar_datasets = glob.glob("madlad-400/data/ar/ar_clean_*")
dsir = HashedNgramDSIR(
    ar_datasets,
    ["madlad-400-ar-sample.jsonl"],
    cache_dir=".cache/dsir",
)
dsir.fit_importance_estimator(num_tokens_to_fit= "**********"
dsir.compute_importance_weights()
dsir.resample(
    out_dir="madlad-ar-sampled",
    num_to_sample=5000000,
    cache_dir=".cache/resampled",
)

# Sample EN datasets at a much lower ratio, just to help maintain base model capabilities.
logger.info("Sampling EN datasets...")
with open("madlad-400-en-sample.jsonl", "a+") as outfile:
    ...
en_datasets = glob.glob("madlad-400/data/en/en_clean_*")
dsir = HashedNgramDSIR(
    en_datasets,
    ["madlad-400-en-sample.jsonl"],
    cache_dir=".cache/dsir-en",
)
dsir.fit_importance_estimator(num_tokens_to_fit= "**********"
dsir.compute_importance_weights()
dsir.resample(
    out_dir="madlad-en-sampled",
    num_to_sample=500000,
    cache_dir=".cache/resampled-en",
)

# Load and unify the various EN/AR files.
logger.info("Unifying dataset...")
sample_files = list(glob.glob("madlad-ar-sampled/*.jsonl")) + list(
    glob.glob("madlad-en-sampled/*.jsonl")
)
all_datasets = []
for path in sample_files:
    if os.stat(path).st_size:
        all_datasets.append(datasets.Dataset.from_json(path))

# Combine everything.
datasets.concatenate_datasets(all_datasets).shuffle(seed=42).to_parquet(
    "madlad-pretrain-sample-combined.parquet"
)