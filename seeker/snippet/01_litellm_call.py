#date: 2025-08-18T17:08:42Z
#url: https://api.github.com/gists/c8303b598c5cc0f64fb0f2d196750a9e
#owner: https://api.github.com/users/grahama1970

#!/usr/bin/env python3

"""
LiteLLM Call - Easy async LLM batch runner with automatic image support

WHAT IT DOES:
- Run multiple LLM prompts in parallel for speed
- Automatically detects and includes images from URLs or local files
- Works with any LiteLLM-supported model (OpenAI, Anthropic, Ollama, etc.)
- Handles all image processing automatically (compression, base64 encoding)

QUICK START:
1. Basic text prompt:
   $ python litellm_call.py "What is 2+2?"

2. Multiple prompts (run in parallel):
   $ python litellm_call.py "What is 2+2?" "What is the capital of France?"

3. Prompt with images (auto-detected):
   $ python litellm_call.py "What's in this image? /path/to/image.jpg"
   $ python litellm_call.py "Compare: https://example.com/cat.jpg and dog.png"

4. From files:
   $ python litellm_call.py @prompts.txt        # One prompt per line
   $ python litellm_call.py prompts.json        # JSON array of prompts
   $ python litellm_call.py @prompts.jsonl      # JSON Lines format
   
5. From stdin:
   $ echo "What is 2+2?" | python litellm_call.py --stdin
   $ cat prompts.jsonl | python litellm_call.py --stdin --jsonl

ENVIRONMENT SETUP:
- OLLAMA_DEFAULT_MODEL: Model to use (default: "ollama/gemma3:12b")
- OLLAMA_BASE_URL: API endpoint (default: "http://localhost:11434")
- OLLAMA_API_KEY: API key if required

ADVANCED USAGE:
- Override model: --model "gpt-4"
- Custom API: --api-base "https://api.openai.com/v1"
- With API key: --api-key "sk-..."

INPUT FORMATS:
1. Simple string: "What is 2+2?"
2. With image: {"text": "Explain this", "image": "path/to/image.jpg"}
3. Full control: {"model": "gpt-4", "messages": [...], "temperature": 0.7}

FEATURES:
- Automatic image detection in prompts (URLs and file paths)
- Smart image compression to stay under API limits
- Parallel processing with progress bar
- Automatic retries on failures
- Silent handling of missing/broken images
- Supports all common image formats (jpg, png, gif, etc.)

"""
import asyncio
import sys
import json
import base64
import io
import os
import re
from pathlib import Path
from typing import List, Tuple, Any, Dict

import httpx
from PIL import Image
from litellm import acompletion
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm
from loguru import logger
from dotenv import load_dotenv, find_dotenv
from urlextract import URLExtract
import typer

logger.remove()
logger.add(sys.stderr, level="WARNING")
from lean4_prover.utils.litellm_cache import initialize_litellm_cache

load_dotenv(find_dotenv())

# -----------------------------------------------------------------------------
#  Typer app  (NEW)
# -----------------------------------------------------------------------------
cli = typer.Typer(
    name="litellm_call",
    help="Fast async LLM batch runner with inline image support via LiteLLM / Ollama.",
)

MODEL = os.getenv("OLLAMA_DEFAULT_MODEL", "ollama/gemma3:12b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "")

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
extractor = URLExtract()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def safe_image(path: Path) -> bool:
    """True if file exists, has an image extension, and PIL can open it."""
    try:
        return path.exists() and path.suffix.lower() in IMAGE_EXT and Image.open(path).verify() is None
    except Exception:
        return False


def extract_images(text: str) -> tuple[List[str], str]:
    """
    Return:
        - list[str] of all valid image URLs/paths (remote & local)
        - cleaned prompt text with placeholders {Image 1}, {Image 2}, …
    """
    found, seen = [], set()

    # 1) Strip XML/HTML tags ----------------------------------------------------
    plain = re.sub(r"<[^>]+>", "", text)

    # 2) Remote URLs -----------------------------------------------------------
    for url in extractor.find_urls(plain):
        url = url.strip()
        if url.lower().endswith(tuple(IMAGE_EXT)) and url not in seen:
            found.append(url)
            seen.add(url)

    # 3) Local files -----------------------------------------------------------
    tokens = re.findall(r'(?: "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"f "**********"o "**********"r "**********"  "**********"t "**********"o "**********"k "**********"  "**********"i "**********"n "**********"  "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********": "**********"
        tok = tok.strip('"\'')
        if not tok:
            continue
        candidate = Path(tok).expanduser().resolve()
        if safe_image(candidate) and str(candidate) not in seen:
            found.append(str(candidate))
            seen.add(str(candidate))

    # 4) Build cleaned prompt with placeholders --------------------------------
    cleaned = text
    for idx, img in enumerate(found, 1):
        placeholder = f"{{Image {idx}}}"
        cleaned = cleaned.replace(img, placeholder)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    return found, cleaned


def compress_image(path_str: str, max_kb: int = 1000) -> str:
    """Return base-64 data-URI for a *local* image, compressed if required."""
    path = Path(path_str)
    img_bytes = path.read_bytes()
    max_bytes = max_kb * 1024

    if len(img_bytes) <= max_bytes:
        mime = f"image/{path.suffix[1:]}"
        return f"data:{mime};base64,{base64.b64encode(img_bytes).decode()}"

    img = Image.open(io.BytesIO(img_bytes))
    quality = 85
    while quality > 20:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        if len(buf.getvalue()) <= max_bytes:
            return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"
        quality -= 10

    img.thumbnail((img.width // 2, img.height // 2))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=30)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def fetch_remote_image(url: str) -> str | None:
    """Download remote image and return base-64 data-URI or None on failure."""
    try:
        r = httpx.get(url, timeout=10)
        r.raise_for_status()
        mime = r.headers.get("content-type", "image/jpeg").split(";")[0]
        return f"data:{mime};base64,{base64.b64encode(r.content).decode()}"
    except Exception as e:
        logger.warning(f"Skipping remote image {url}: {e}")
        return None


# -----------------------------------------------------------------------------
# LLM call with retry
# -----------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=4))
async def _call(params: Dict[str, Any], idx: int) -> Tuple[int, str]:
    resp = await acompletion(**params)
    return idx, resp.choices[0].message.content


# -----------------------------------------------------------------------------
# Batch runner
# -----------------------------------------------------------------------------

async def litellm_call(prompts: List[Any]) -> List[str]:
    """Run prompts: strings, dicts, or raw LiteLLM dicts."""
    
    # Accept a single prompt as well
    if isinstance(prompts, (str, dict)):
        prompts = [prompts]
    
    tasks: List[asyncio.Task] = []

    for idx, item in enumerate(prompts):
        # Raw LiteLLM dict
        if isinstance(item, dict) and "messages" in item:
            defaults = {
                "model": MODEL,
                "api_base": OLLAMA_BASE_URL,
                "api_key": OLLAMA_API_KEY,
            }
            params = {**defaults, **item}  # user keys override env
            tasks.append(asyncio.create_task(_call(params, idx)))
            continue

        # Parse text & images
        if isinstance(item, dict):
            text = str(item.get("text", ""))
            images = [str(item["image"])] if "image" in item else []
        else:
            images, text = extract_images(str(item))  # returns (list[str], cleaned)

        # Build OpenAI-style content
        content_parts: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        for img in images:
            if img.startswith("http"):
                url = fetch_remote_image(img)
                if url is None:
                    continue  # skip unreachable
            else:
                url = compress_image(img)
            content_parts.append({"type": "image_url", "image_url": {"url": url}})

        params = {
            "model": MODEL,
            "api_base": OLLAMA_BASE_URL,
            "api_key": OLLAMA_API_KEY,
            "messages": [{"role": "user", "content": content_parts}],
        }
        tasks.append(asyncio.create_task(_call(params, idx)))

    results = [None] * len(tasks)
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing"):
        idx, answer = await coro
        results[idx] = answer
        logger.info(f"\nQ{idx}: {str(prompts[idx])[:50]}...\nA{idx}: {answer[:100]}...")

    return results


# -----------------------------------------------------------------------------
# Typer Cli
# -----------------------------------------------------------------------------
# @cli.callback()
# def main():
#     pass

@cli.command()
def main(
    sources: List[str] = typer.Argument(None, help="…"),
    model: str = typer.Option(MODEL, "--model", "-m", help="…"),
    api_base: str = typer.Option(OLLAMA_BASE_URL, "--api-base", help="…"),
    api_key: str = typer.Option(OLLAMA_API_KEY, "--api-key", help="…"),
    stdin: bool = typer.Option(False, "--stdin", help="…"),
    jsonl: bool = typer.Option(False, "--jsonl", help="…"),
):
    """
    Run any combination of prompts via LiteLLM.

    Examples\n
    --------\n
    litellm_call run "What is 2+2?"\n
    litellm_call run @questions.txt\n
    litellm_call run prompts.json\n
    cat lines.jsonl | litellm_call run --stdin --jsonl\n
    """
    print(">>> typer sources =", sources)
    prompts: List[Any] = []

    # 1) STDIN
    if stdin or (sources == ["-"]):
        for line in sys.stdin:
            line = line.rstrip("\n")
            if jsonl:
                prompts.append(json.loads(line))
            else:
                prompts.append(line)

    # 2) Positional sources
    for src in sources or []:
        if src == "-":
            continue  # already handled above

        # @file expansion
        if src.startswith("@"):
            src = src[1:]
        path = Path(src)

        if not path.exists():
            # Treat literal string
            prompts.append(src)
            continue

        # Decide how to parse the file
        if path.suffix.lower() == ".json":
            prompts.extend(json.loads(path.read_text()))
        elif path.suffix.lower() == ".jsonl" or jsonl:
            prompts.extend(json.loads(l) for l in path.read_text().splitlines() if l.strip())
        else:
            prompts.extend(path.read_text().splitlines())

    if not prompts:
        typer.echo("No prompts provided.", err=True)
        raise typer.Exit(1)
    
    # TEMPORARY DEBUG
    print("*** PROMPT TO MODEL:", prompts)
    
    results = asyncio.run(litellm_call(prompts))
    for r in results:
        typer.echo(r)

# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

def debug() -> None:
    prompts = [
        "What is the capital of France?",
        "Calculate 15+27+38",
        "What is 3 + 5? Return JSON: {question:string,answer:number}",
        "What is this animal eating? proof_of_concept/ollama_turbo/images/image2.png",
        "Describe https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Labrador_Retriever_portrait.jpg/960px-Labrador_Retriever_portrait.jpg  and https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/960px-Cat_November_2010-1a.jpg",
        {"text": "Explain this meme", "image": "proof_of_concept/ollama_turbo/images/image.png"},
        {
            "model": "ollama/gpt-oss:120b",
            "api_base": "https://ollama.com",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Tell me a short joke."}
            ],
            "temperature": 1.0
        }
    ]

    results = asyncio.run(litellm_call(prompts))
    print("\nFinal Results:")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r}")


DEBUG=False
if __name__ == "__main__":
    # print(f"DEBUG -- sys.argv received by script: {sys.argv}") # <-- ADD THIS LINE
    debug() if DEBUG else cli()