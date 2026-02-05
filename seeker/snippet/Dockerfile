#date: 2026-02-05T17:40:03Z
#url: https://api.github.com/gists/9610a93371a294afebc20cb8698825dc
#owner: https://api.github.com/users/renoirb

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install whisper
RUN pip install --no-cache-dir openai-whisper

# Set working directory
WORKDIR /workspace

# Default command
CMD ["whisper", "--help"]
