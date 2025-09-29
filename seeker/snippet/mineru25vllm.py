#date: 2025-09-29T16:55:18Z
#url: https://api.github.com/gists/632d96f3c8a002fdabbfe28e9083dce8
#owner: https://api.github.com/users/chandradeepc

import modal
from typing import Dict, Any, Optional
from pydantic import BaseModel
import json
import os
import uuid
import base64
import io

# Define the image - using sglang base image as recommended by mineru
image = (
    modal.Image.from_registry(
        "vllm/vllm-openai:v0.10.1.1",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster model transfers
        }
    )
    .run_commands("ln -s /usr/bin/python3 /usr/bin/python")
    .entrypoint([])
    .apt_install(
        [
            "libgl1",
            "fonts-noto-core",
            "fonts-noto-cjk",
            "fontconfig",
        ]
    )
    .run_commands(
        [
            "fc-cache -fv",
            "apt-get clean",
            "rm -rf /var/lib/apt/lists/*",
        ]
    )
    .pip_install(
        [
            "setuptools<70.0.0",  # Fix setuptools compatibility issue with antlr4
            "transformers>=4.56.0",  # Update transformers to latest version for mineru2_qwen support
            "mineru[core]",  # Install only mineru[core] to avoid dependency conflicts
            "nest-asyncio",  # Allow nested event loops to fix "This event loop is already running" error
            "Pillow",  # For image format conversion
        ]
    )
)

# Create the Modal app
app = modal.App("mineru-vllm")


class PDFRequest(BaseModel):
    file: str  # base64 encoded PDF
    lang: str = "en"
    start_page: int = 0
    end_page: Optional[int] = None


class PDFResponse(BaseModel):
    result: Dict[str, Any]
    status: str = "success"


class PDFErrorResponse(BaseModel):
    error: str
    status: str = "failed"


# Global variable to store initialization status
initialized = False


def add_base64_images_to_json(json_data, image_dir):
    """
    Recursively traverse the JSON structure and add base64 image content
    to spans with type="image" and image_path property.
    Converts images to PNG format before base64 encoding.
    """
    import os
    from loguru import logger
    from PIL import Image

    def process_item(item):
        if isinstance(item, dict):
            # Check if this is an image span
            if item.get("type") == "image" and "image_path" in item:
                image_path = item["image_path"]
                full_image_path = os.path.join(image_dir, image_path)

                try:
                    # Read the image file, convert to PNG, and encode to base64
                    if os.path.exists(full_image_path):
                        # Open image with PIL and convert to PNG
                        with Image.open(full_image_path) as img:
                            # Convert to RGB if necessary (some formats like P mode need this)
                            if img.mode in ("RGBA", "LA", "P"):
                                img = img.convert("RGBA")
                            elif img.mode != "RGB":
                                img = img.convert("RGB")

                            # Save as PNG to BytesIO buffer
                            png_buffer = io.BytesIO()
                            img.save(png_buffer, format="PNG")
                            png_data = png_buffer.getvalue()

                            # Encode to base64
                            base64_image = base64.b64encode(png_data).decode("utf-8")
                            item["base64image"] = base64_image
                            # logger.info(f"Added base64 PNG image for {image_path}")
                    else:
                        pass  # logger.warning(f"Image file not found: {full_image_path}")
                except Exception as e:
                    pass  # logger.error(f"Error processing image {image_path}: {str(e)}")

            # Recursively process all dictionary values
            for value in item.values():
                process_item(value)

        elif isinstance(item, list):
            # Recursively process all list items
            for list_item in item:
                process_item(list_item)

    process_item(json_data)


def load_model():
    """Initialize MinerU with vllm backend"""
    return {"status": "ready", "backend": "vlm-vllm-async-engine"}


@app.function(
    image=image,
    # gpu=["T4", "L4", "A10", "L40S"],
    gpu=["L40S", "A100-80GB", "H100", "H200"],
    timeout=1800,  # 30 minutes
    min_containers=0,
    max_containers=10,
    scaledown_window=60,  # 1 minute
    enable_memory_snapshot=True,
    experimental_options={"enable_gpu_snapshot": False},
)
@modal.fastapi_endpoint(method="POST")
async def parse_pdf(request: PDFRequest):
    """
    Parse PDF using MinerU with vllm backend

    Returns the JSON result directly instead of saving to file
    """
    # Apply nest_asyncio to allow nested event loops - must be done in Modal runtime
    import nest_asyncio

    nest_asyncio.apply()

    # Initialize on first use
    global initialized
    if not initialized:
        load_model()
        initialized = True

    # Import MinerU components (only available in remote environment)
    from mineru.cli.common import (
        convert_pdf_bytes_to_bytes_by_pypdfium2,
        prepare_env,
        read_fn,
    )
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
    from loguru import logger
    import multiprocessing as mp
    import shutil

    file = request.file
    lang = request.lang
    start_page = request.start_page
    end_page = request.end_page

    if not file:
        return PDFErrorResponse(error="Please provide a PDF file (base64) to parse")

    # Decode base64 to bytes
    try:
        file_bytes = base64.b64decode(file)
    except Exception as e:
        return PDFErrorResponse(error=f"Invalid base64 file data: {str(e)}")

    # Validate it's a PDF
    if not file_bytes.startswith(b"%PDF"):
        return PDFErrorResponse(error="File does not appear to be a valid PDF")

    try:
        # Create temporary directories
        temp_id = str(uuid.uuid4())
        temp_dir = f"/tmp/mineru_{temp_id}"
        output_dir = f"{temp_dir}/output"
        os.makedirs(output_dir, exist_ok=True)

        # Use the decoded bytes directly - no need to save and read back
        pdf_bytes = file_bytes
        pdf_file_name = "document"

        # Convert PDF pages if needed
        if start_page > 0 or end_page is not None:
            pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                pdf_bytes, start_page, end_page
            )

        # Prepare environment
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, "vlm")
        image_writer = FileBasedDataWriter(local_image_dir)
        md_writer = FileBasedDataWriter(local_md_dir)

        # Use sglang-engine backend for analysis
        backend = "vllm-async-engine"
        # logger.info(f"Starting PDF analysis with backend: {backend}")

        # Analyze document with VLM
        # logger.info("About to call vlm_doc_analyze...")
        middle_json, infer_result = vlm_doc_analyze(
            pdf_bytes, image_writer=image_writer, backend=backend, server_url=None
        )

        # Add base64 image content to image spans
        # logger.info("Adding base64 image content to image spans...")
        add_base64_images_to_json(middle_json, local_image_dir)

        # Instead of saving to file, return the JSON directly
        return PDFResponse(result=middle_json)

    except Exception as e:
        # logger.exception(f"Error processing PDF: {str(e)}")
        return PDFErrorResponse(error=f"Failed to process PDF: {str(e)}")

    finally:
        # Cleanup temporary files
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception as cleanup_error:
            pass  # logger.warning(f"Failed to cleanup temporary files: {cleanup_error}")


# To run the ephemeral endpoint: modal serve minerusglangapp.py
# To deploy: modal deploy minerusglangapp.py