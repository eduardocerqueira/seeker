#date: 2025-07-09T16:50:29Z
#url: https://api.github.com/gists/a70123b78193154d821814e45a322bfb
#owner: https://api.github.com/users/djb4ai

#!/usr/bin/env python3
"""
Groq Batch OCR Processing Script

This script generates JSONL batch files for Groq's Batch API to process
hotel invoice PDFs asynchronously with 50% cost savings.

Based on: https://console.groq.com/docs/batch
"""

from groq import Groq
import fitz  # PyMuPDF
import io
import os
from PIL import Image
import base64
import json
import sys
from datetime import datetime

# Get API key from environment variable
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    print("Error: GROQ_API_KEY environment variable not set")
    print("Please set your Groq API key:")
    print("  export GROQ_API_KEY='your_api_key_here'")
    sys.exit(1)

client = Groq(api_key=api_key)


def encode_image(image_path):
    """Encode image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def pdf_to_base64_images(pdf_path):
    """Convert PDF pages to base64 encoded images."""
    try:
        pdf_document = fitz.open(pdf_path)
        base64_images = []
        temp_image_paths = []

        total_pages = len(pdf_document)
        print(f"  Processing {total_pages} pages from {os.path.basename(pdf_path)}")

        for page_num in range(total_pages):
            page = pdf_document.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.open(io.BytesIO(pix.tobytes()))
            temp_image_path = f"temp_page_{page_num}_{os.path.basename(pdf_path)}.png"
            img.save(temp_image_path, format="PNG")
            temp_image_paths.append(temp_image_path)
            
            base64_image = encode_image(temp_image_path)
            if base64_image:
                base64_images.append(base64_image)
            else:
                print(f"  Failed to encode page {page_num}")

        # Clean up temporary files
        for temp_image_path in temp_image_paths:
            try:
                os.remove(temp_image_path)
            except Exception as e:
                print(f"  Warning: Could not remove temp file {temp_image_path}: {e}")

        pdf_document.close()
        return base64_images
    except Exception as e:
        print(f"  Error processing PDF {pdf_path}: {e}")
        return []


def create_batch_request(custom_id, base64_image, model="llama-3.3-70b-versatile"):
    """Create a single batch request for invoice data extraction."""
    system_prompt = """You are an OCR-like data extraction tool that extracts hotel invoice data from PDFs.

1. Please extract the data in this hotel invoice, grouping data according to theme/sub groups, and then output into JSON.

2. Please keep the keys and values of the JSON in the original language.

3. The type of data you might encounter in the invoice includes but is not limited to: hotel information, guest information, invoice information, room charges, taxes, and total charges etc.

4. If the page contains no charge data, please output an empty JSON object and don't make up any data.

5. If there are blank data fields in the invoice, please include them as "null" values in the JSON object.

6. If there are tables in the invoice, capture all of the rows and columns in the JSON object. Even if a column is blank, include it as a key in the JSON object with a null value.

7. If a row is blank denote missing fields with "null" values.

8. Don't interpolate or make up data.

9. Please maintain the table structure of the charges, i.e. capture all of the rows and columns in the JSON object."""

    batch_request = {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract the data in this hotel invoice and output into JSON"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}
                    ]
                }
            ],
            "temperature": 0.0
        }
    }
    return batch_request


def generate_batch_file(read_path, batch_file_path, model="llama-3.3-70b-versatile"):
    """Generate JSONL batch file from PDFs in the specified directory."""
    if not os.path.exists(read_path):
        print(f"Error: Input directory '{read_path}' does not exist")
        return False

    # Get list of PDF files
    pdf_files = [f for f in os.listdir(read_path) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in '{read_path}'")
        return False

    print(f"Found {len(pdf_files)} PDF files to process")
    
    batch_requests = []
    request_count = 0

    for filename in pdf_files:
        file_path = os.path.join(read_path, filename)
        print(f"\nProcessing: {filename}")
        
        base64_images = pdf_to_base64_images(file_path)
        if not base64_images:
            print(f"  Skipping {filename} - failed to process")
            continue
            
        # Create batch requests for each page
        for page_num, base64_image in enumerate(base64_images):
            custom_id = f"{filename}_page_{page_num + 1}"
            batch_request = create_batch_request(custom_id, base64_image, model)
            batch_requests.append(batch_request)
            request_count += 1

    if not batch_requests:
        print("No valid requests generated")
        return False

    # Write batch file
    try:
        os.makedirs(os.path.dirname(batch_file_path), exist_ok=True)
        with open(batch_file_path, 'w', encoding='utf-8') as f:
            for request in batch_requests:
                f.write(json.dumps(request, ensure_ascii=False) + '\n')
        
        print(f"\n✓ Generated batch file: {batch_file_path}")
        print(f"✓ Total requests: {request_count}")
        print(f"✓ Total files processed: {len([f for f in pdf_files if any(r['custom_id'].startswith(f) for r in batch_requests)])}")
        
        # Create metadata file
        metadata = {
            "created_at": datetime.now().isoformat(),
            "total_requests": request_count,
            "total_files": len(pdf_files),
            "model": model,
            "source_directory": read_path
        }
        
        metadata_path = batch_file_path.replace('.jsonl', '_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Generated metadata file: {metadata_path}")
        return True
        
    except Exception as e:
        print(f"Error writing batch file: {e}")
        return False


def submit_batch(batch_file_path, completion_window="24h"):
    """Submit batch file to Groq Batch API."""
    try:
        print(f"Uploading batch file: {batch_file_path}")
        
        # Upload file
        with open(batch_file_path, "rb") as f:
            file_response = client.files.create(file=f, purpose="batch")
        
        print(f"✓ File uploaded with ID: {file_response.id}")
        
        # Create batch
        batch_response = client.batches.create(
            completion_window=completion_window,
            endpoint="/v1/chat/completions",
            input_file_id=file_response.id,
        )
        
        print(f"✓ Batch created with ID: {batch_response.id}")
        print(f"✓ Status: {batch_response.status}")
        print(f"✓ Expires at: {datetime.fromtimestamp(batch_response.expires_at)}")
        
        # Save batch info
        batch_info = {
            "batch_id": batch_response.id,
            "file_id": file_response.id,
            "status": batch_response.status,
            "created_at": datetime.fromtimestamp(batch_response.created_at).isoformat(),
            "expires_at": datetime.fromtimestamp(batch_response.expires_at).isoformat(),
            "completion_window": completion_window,
            "batch_file_path": batch_file_path
        }
        
        batch_info_path = batch_file_path.replace('.jsonl', '_batch_info.json')
        with open(batch_info_path, 'w', encoding='utf-8') as f:
            json.dump(batch_info, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Batch info saved to: {batch_info_path}")
        return batch_response.id
        
    except Exception as e:
        print(f"Error submitting batch: {e}")
        return None


def check_batch_status(batch_id):
    """Check the status of a batch job."""
    try:
        response = client.batches.retrieve(batch_id)
        
        print(f"Batch ID: {response.id}")
        print(f"Status: {response.status}")
        print(f"Created: {datetime.fromtimestamp(response.created_at)}")
        print(f"Expires: {datetime.fromtimestamp(response.expires_at)}")
        
        if response.request_counts:
            counts = response.request_counts
            print(f"Requests - Total: {counts.total}, Completed: {counts.completed}, Failed: {counts.failed}")
        
        if response.status == "completed":
            print(f"Output file ID: {response.output_file_id}")
            if response.error_file_id:
                print(f"Error file ID: {response.error_file_id}")
        
        return response
        
    except Exception as e:
        print(f"Error checking batch status: {e}")
        return None


def retrieve_batch_results(batch_id, output_path):
    """Retrieve and save batch results."""
    try:
        # Get batch info
        batch = client.batches.retrieve(batch_id)
        
        if batch.status != "completed":
            print(f"Batch not completed yet. Status: {batch.status}")
            return False
        
        if not batch.output_file_id:
            print("No output file available")
            return False
        
        # Download results
        print(f"Downloading results from file ID: {batch.output_file_id}")
        response = client.files.content(batch.output_file_id)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        response.write_to_file(output_path)
        
        print(f"✓ Results saved to: {output_path}")
        
        # Download errors if available
        if batch.error_file_id:
            error_path = output_path.replace('.jsonl', '_errors.jsonl')
            error_response = client.files.content(batch.error_file_id)
            error_response.write_to_file(error_path)
            print(f"✓ Errors saved to: {error_path}")
        
        return True
        
    except Exception as e:
        print(f"Error retrieving batch results: {e}")
        return False


def main():
    """Main function with command-line interface."""
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python groq-batch-ocr.py generate <input_dir> [output_batch_file] [model]")
        print("  python groq-batch-ocr.py submit <batch_file> [completion_window]")
        print("  python groq-batch-ocr.py status <batch_id>")
        print("  python groq-batch-ocr.py retrieve <batch_id> <output_file>")
        print("  python groq-batch-ocr.py list")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "generate":
        if len(sys.argv) < 3:
            print("Error: Please specify input directory")
            sys.exit(1)
        
        input_dir = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else "./batch_files/invoice_batch.jsonl"
        model = sys.argv[4] if len(sys.argv) > 4 else "llama-3.3-70b-versatile"
        
        success = generate_batch_file(input_dir, output_file, model)
        if success:
            print(f"\nNext steps:")
            print(f"1. Submit batch: python groq-batch-ocr.py submit {output_file}")
            print(f"2. Check status: python groq-batch-ocr.py status <batch_id>")
            print(f"3. Retrieve results: python groq-batch-ocr.py retrieve <batch_id> ./results/batch_results.jsonl")
    
    elif command == "submit":
        if len(sys.argv) < 3:
            print("Error: Please specify batch file")
            sys.exit(1)
        
        batch_file = sys.argv[2]
        completion_window = sys.argv[3] if len(sys.argv) > 3 else "24h"
        
        batch_id = submit_batch(batch_file, completion_window)
        if batch_id:
            print(f"\nBatch submitted successfully!")
            print(f"Batch ID: {batch_id}")
            print(f"Check status with: python groq-batch-ocr.py status {batch_id}")
    
    elif command == "status":
        if len(sys.argv) < 3:
            print("Error: Please specify batch ID")
            sys.exit(1)
        
        batch_id = sys.argv[2]
        check_batch_status(batch_id)
    
    elif command == "retrieve":
        if len(sys.argv) < 4:
            print("Error: Please specify batch ID and output file")
            sys.exit(1)
        
        batch_id = sys.argv[2]
        output_file = sys.argv[3]
        
        success = retrieve_batch_results(batch_id, output_file)
        if success:
            print("Results retrieved successfully!")
    
    elif command == "list":
        try:
            response = client.batches.list()
            print("All batch jobs:")
            for batch in response.data:
                print(f"  {batch.id} - {batch.status} - Created: {datetime.fromtimestamp(batch.created_at)}")
        except Exception as e:
            print(f"Error listing batches: {e}")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
