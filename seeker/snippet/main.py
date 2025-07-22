#date: 2025-07-22T16:59:19Z
#url: https://api.github.com/gists/59a3bb9b23098b242232667813fa0c1c
#owner: https://api.github.com/users/davanstrien

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "datasets",
#     "huggingface-hub",
#     "pillow",
#     "vllm",
#     "tqdm",
#     "toolz",
# ]
# ///

"""
Convert document images to markdown using Nanonets-OCR-s with vLLM.

This script processes images through the Nanonets-OCR-s model to extract
text and structure as markdown, ideal for document understanding tasks.
"""

import argparse
import base64
import io
import logging
import os
import sys
from typing import List, Dict, Any, Union

from PIL import Image
from datasets import load_dataset
from huggingface_hub import login
from toolz import partition_all
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_ocr_message(
    image: Union[Image.Image, Dict[str, Any], str],
    prompt: str = "Convert this image to markdown. Include all text, tables, equations, and structure.",
) -> List[Dict]:
    """Create chat message for OCR processing."""
    # Convert to PIL Image if needed
    if isinstance(image, Image.Image):
        pil_img = image
    elif isinstance(image, dict) and "bytes" in image:
        pil_img = Image.open(io.BytesIO(image["bytes"]))
    elif isinstance(image, str):
        pil_img = Image.open(image)
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")
    
    # Convert to base64 data URI
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    data_uri = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
    
    # Return message in vLLM format
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def main(
    input_dataset: str,
    output_dataset: str,
    image_column: str = "image",
    batch_size: int = 8,
    model: str = "nanonets/Nanonets-OCR-s",
    max_model_len: int = 8192,
    max_tokens: "**********"
    gpu_memory_utilization: float = 0.7,
    hf_token: "**********"
    split: str = "train",
    max_samples: int = None,
    private: bool = False,
):
    """Process images from HF dataset through OCR model."""
    
    # Login to HF if token provided
    HF_TOKEN = "**********"
 "**********"  "**********"  "**********"  "**********"  "**********"i "**********"f "**********"  "**********"H "**********"F "**********"_ "**********"T "**********"O "**********"K "**********"E "**********"N "**********": "**********"
        login(token= "**********"
    
    # Load dataset
    logger.info(f"Loading dataset: {input_dataset}")
    dataset = load_dataset(input_dataset, split=split)
    
    # Validate image column
    if image_column not in dataset.column_names:
        raise ValueError(f"Column '{image_column}' not found. Available: {dataset.column_names}")
    
    # Limit samples if requested
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
        logger.info(f"Limited to {len(dataset)} samples")
    
    # Initialize vLLM
    logger.info(f"Initializing vLLM with model: {model}")
    llm = LLM(
        model=model,
        trust_remote_code=True,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        limit_mm_per_prompt={"image": 1},
    )
    
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for OCR
        max_tokens= "**********"
    )
    
    # Process images in batches
    all_markdown = []
    
    logger.info(f"Processing {len(dataset)} images in batches of {batch_size}")
    
    # Process in batches to avoid memory issues
    for batch_indices in tqdm(
        partition_all(batch_size, range(len(dataset))),
        total=(len(dataset) + batch_size - 1) // batch_size,
        desc="OCR processing"
    ):
        batch_indices = list(batch_indices)
        batch_images = [dataset[i][image_column] for i in batch_indices]
        
        try:
            # Create messages for batch
            batch_messages = [make_ocr_message(img) for img in batch_images]
            
            # Process with vLLM
            outputs = llm.chat(batch_messages, sampling_params)
            
            # Extract markdown from outputs
            for output in outputs:
                markdown_text = output.outputs[0].text.strip()
                all_markdown.append(markdown_text)
                
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            # Add error placeholders for failed batch
            all_markdown.extend(["[OCR FAILED]"] * len(batch_images))
    
    # Add markdown column to dataset
    logger.info("Adding markdown column to dataset")
    dataset = dataset.add_column("markdown", all_markdown)
    
    # Push to hub
    logger.info(f"Pushing to {output_dataset}")
    dataset.push_to_hub(output_dataset, private= "**********"=HF_TOKEN)
    
    logger.info("✅ OCR conversion complete!")
    logger.info(f"Dataset available at: https://huggingface.co/datasets/{output_dataset}")


if __name__ == "__main__":
    # Show example usage if no arguments
    if len(sys.argv) == 1:
        print("=" * 80)
        print("Nanonets OCR to Markdown Converter")
        print("=" * 80)
        print("\nThis script converts document images to structured markdown using")
        print("the Nanonets-OCR-s model with vLLM acceleration.")
        print("\nFeatures:")
        print("- LaTeX equation recognition")
        print("- Table extraction and formatting")
        print("- Document structure preservation")
        print("- Signature and watermark detection")
        print("\nExample usage:")
        print("\n1. Basic OCR conversion:")
        print("   uv run main.py document-images markdown-docs")
        print("\n2. With custom settings:")
        print("   uv run main.py scanned-pdfs extracted-text \\")
        print("       --image-column page \\")
        print("       --batch-size 16 \\")
        print("       --gpu-memory-utilization 0.8")
        print("\n3. Running on HF Jobs:")
        print("   hfjobs run \\")
        print("     --flavor l4x1 \\")
        print("     --secret HF_TOKEN= "**********"
        print("     ghcr.io/astral-sh/uv:latest \\")
        print("     /bin/bash -c \"")
        print("       uv run https://huggingface.co/datasets/davanstrien/dataset-creation-scripts/raw/main/ocr-vllm/main.py \\\\")
        print("         your-document-dataset \\\\")
        print("         your-markdown-output \\\\")
        print("         --batch-size 32")
        print("     \"")
        print("\n" + "=" * 80)
        print("\nFor full help, run: uv run main.py --help")
        sys.exit(0)
    
    parser = argparse.ArgumentParser(
        description="OCR images to markdown using Nanonets-OCR-s",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  uv run main.py my-images-dataset ocr-results

  # With specific image column
  uv run main.py documents extracted-text --image-column scan

  # Process subset for testing
  uv run main.py large-dataset test-output --max-samples 100
        """
    )
    
    parser.add_argument(
        "input_dataset",
        help="Input dataset ID from Hugging Face Hub"
    )
    parser.add_argument(
        "output_dataset",
        help="Output dataset ID for Hugging Face Hub"
    )
    parser.add_argument(
        "--image-column",
        default="image",
        help="Column containing images (default: image)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for processing (default: 8)"
    )
    parser.add_argument(
        "--model",
        default="nanonets/Nanonets-OCR-s",
        help="Model to use (default: nanonets/Nanonets-OCR-s)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum model context length (default: 8192)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate (default: "**********"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization (default: 0.7)"
    )
    parser.add_argument(
        "--hf-token",
        help= "**********"
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make output dataset private"
    )
    
    args = parser.parse_args()
    
    main(
        input_dataset=args.input_dataset,
        output_dataset=args.output_dataset,
        image_column=args.image_column,
        batch_size=args.batch_size,
        model=args.model,
        max_model_len=args.max_model_len,
        max_tokens= "**********"
        gpu_memory_utilization=args.gpu_memory_utilization,
        hf_token= "**********"
        split=args.split,
        max_samples=args.max_samples,
        private=args.private,
    )