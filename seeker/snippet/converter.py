#date: 2025-11-27T16:52:18Z
#url: https://api.github.com/gists/222789fbf11955931bd3ce91156655a9
#owner: https://api.github.com/users/benerone

import argparse
import subprocess
import sys
import re
import base64
import io
from pathlib import Path
from PIL import Image
import markdown

def run_ollama_ocr(image_path_str, model="deepseek-ocr"):
    """
    Executes the ollama command and returns the output.
    """
    image_path = Path(image_path_str)

    # Construct the prompt exactly as requested in Spec.md and user's feedback
    prompt_text = f"\"./{image_path.as_posix()}\\n<|grounding|>Convert the document to markdown.\""
    
    cmd = ["ollama", "run", "deepseek-ocr", prompt_text]

    print(f"Running command: {' '.join(cmd)}")
    
    try:
        # Using subprocess.run to execute the command and capture output
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True  # Will raise a CalledProcessError for non-zero exit codes
        )
        
        print("Successfully executed Ollama command.")
        
        return result.stdout

    except FileNotFoundError:
        print("Error: 'ollama' command not found. Please ensure it is installed and in your PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Ollama command failed with exit code {e.returncode}", file=sys.stderr)
        print(f"Stderr:\n{e.stderr}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def parse_ocr_output(ocr_text):
    """
    Parses the custom XML-like tags from DeepSeek output.
    Format: <|ref|>TYPE<|/ref|><|det|>[[x1, y1, x2, y2]]<|/det|>CONTENT
    """
    # Regex to capture type, coordinates, and content
    # We use DOTALL so . matches newlines in content
    pattern = re.compile(
        r"<\|ref\|>(?P<type>.*?)<\|/ref\|>"
        r"<\|det\|>\[\[(?P<x1>\d+),\s*(?P<y1>\d+),\s*(?P<x2>\d+),\s*(?P<y2>\d+)\]\]<\|/det\|>"
        r"(?P<content>.*?)"
        r"(?=(?:<\|ref\|>|$))", # Lookahead for next tag or end of string
        re.DOTALL
    )

    items = []
    for match in pattern.finditer(ocr_text):
        items.append({
            "type": match.group("type").strip(),
            "bbox": [
                int(match.group("x1")),
                int(match.group("y1")),
                int(match.group("x2")),
                int(match.group("y2"))
            ],
            "content": match.group("content").strip()
        })
    return items

def image_to_base64(pil_image):
    """Converts a PIL Image to a base64 string."""
    buffered = io.BytesIO()
    pil_image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def generate_html(items, original_image_path):
    """
    Reconstructs the document in a single-column HTML format.
    - Crops images for 'image' type.
    - Converts markdown to HTML for text types.
    - Arranges elements in a flow layout suitable for an A4 page.
    """
    path_obj = Path(original_image_path)
    html_elements = []

    try:
        with Image.open(path_obj) as img:
            # Process items sequentially
            for item in items:
                item_type = item['type']
                element_html = ""
                
                if item_type == "image":
                    img_width, img_height = img.size
                    x1_model, y1_model, x2_model, y2_model = item['bbox']
                    
                    x1 = int(x1_model / 1000 * img_width)
                    y1 = int(y1_model / 1000 * img_height)
                    x2 = int(x2_model / 1000 * img_width)
                    y2 = int(y2_model / 1000 * img_height)

                    try:
                        cropped_img = img.crop((x1, y1, x2, y2))
                        b64_str = image_to_base64(cropped_img)
                        element_html = f'<img src="data:image/png;base64,{b64_str}" alt="Cropped image segment" style="max-width: 100%; height: auto; margin-bottom: 1em;">'
                    except Exception as e:
                        print(f"Warning: Failed to crop or process image at {item['bbox']} (denormalized to {(x1, y1, x2, y2)}): {e}", file=sys.stderr)
                
                else: # For text, title, etc.
                    content = item['content']
                    if item['type'] in ["text", "sub_title", "title", "table"]:
                        # Replace multiple consecutive dots with a single dot
                        content = re.sub(r'\.{2,}', '.', content)
                    
                    # Convert markdown content to HTML
                    content_html = markdown.markdown(content, extensions=['extra'])
                    # Wrap in a div for semantic grouping and styling
                    element_html = f'<div class="content-block type-{item_type}">{content_html}</div>'

                html_elements.append(element_html)
                
    except FileNotFoundError:
        print(f"Error: Original image file not found at {original_image_path}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during HTML generation: {e}", file=sys.stderr)
        sys.exit(1)

    # A4-inspired HTML structure
    full_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{path_obj.name}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        
        body {{
            background-color: #e0e0e0;
            font-family: 'Roboto', 'Helvetica Neue', Arial, sans-serif;
            color: #212121;
            display: flex;
            justify-content: center;
            margin: 0;
            padding: 20px;
        }}
        .page {{
            background: white;
            width: 21cm;
            min-height: 29.7cm;
            padding: 2cm;
            margin: 1cm 0;
            border: 1px solid #ccc;
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            box-sizing: border-box;
        }}
        h1, h2, h3, h4, h5, h6 {{
            font-weight: 700;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }}
        p, li {{
            line-height: 1.5;
            text-align: justify;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1.5em 0;
            font-size: 0.9em;
        }}
        th, td {{
            border: 1px solid #bdbdbd;
            padding: 10px 14px;
        }}
        th {{
            background-color: #f5f5f5;
            font-weight: 700;
        }}
        img {{
            display: block;
            margin-left: auto;
            margin-right: auto;
        }}

        /* Print-specific styles */
        @media print {{
            body, .page {{
                margin: 0;
                padding: 0;
                box-shadow: none;
                border: none;
                background: white;
            }}
            .page {{
                min-height: 0; /* Let content dictate height */
            }}
        }}
    </style>
</head>
<body>
    <div class="page">
        {''.join(html_elements)}
    </div>
</body>
</html>
    """
    return full_html

def main():
    parser = argparse.ArgumentParser(description="OCR Image to Reconstructed HTML")
    parser.add_argument("image_path", help="Path to the source image file")
    
    args = parser.parse_args()
    image_path = Path(args.image_path)
    
    if not image_path.exists():
        print(f"Error: File {image_path} not found.", file=sys.stderr)
        sys.exit(1)
        
    print(f"Step 1: Running OCR on {image_path}...")
    # Step 1: Call Ollama
    ocr_result = run_ollama_ocr(image_path.as_posix())
    
    # Step 2: Parse and Rebuild
    items = parse_ocr_output(ocr_result)
    
    if not items:
        print("Warning: No structured content found in OCR output. Check model or prompt.", file=sys.stderr)
    
    html_content = generate_html(items, image_path)
    
    # Step 3: Save file
    output_filename = f"{image_path.name}.html"
    try:
        with open(output_filename, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Step 3: Success! Saved to {output_filename}")
    except IOError as e:
        print(f"Error saving file: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()