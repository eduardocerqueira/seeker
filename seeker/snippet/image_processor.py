#date: 2025-09-30T17:09:05Z
#url: https://api.github.com/gists/f1d5da0edfded2a96d8145d213b7f2d3
#owner: https://api.github.com/users/harrisonrw

#!/usr/bin/env python3
"""
Filename: image_processor.py
Author: Robert Harrison
Date: September 30, 2025
Description: Resize image to specified size, convert to circle, and save as PNG with transparency.
Usage: python image_processor.py <input_image_path> <output_image_path> [size] [padding]
       Example: python image_processor.py input.png output.png 44 2
"""

from PIL import Image, ImageDraw, ImageFilter
import sys

def process_image(input_path, output_path, size=42, padding=2):
    """Resize image to specified size, convert to circle, and save as PNG with transparency."""
    try:
        # Open and resize the image to specified size
        with Image.open(input_path) as img:
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            
            # Convert to RGBA to support transparency
            img = img.convert("RGBA")
            
            # Create a new image with transparent background
            output_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            
            # Create a circular mask with soft edges and specified padding
            mask = Image.new("L", (size, size), 0)
            draw = ImageDraw.Draw(mask)
            draw.ellipse((padding, padding, size-padding, size-padding), fill=255)
            
            # Apply a slight blur to soften the edges for better mobile rendering
            mask = mask.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            # Create drop shadow
            shadow_offset = 1
            shadow_blur = 1
            
            # Create shadow mask (slightly offset and blurred)
            shadow_mask = Image.new("L", (size, size), 0)
            shadow_draw = ImageDraw.Draw(shadow_mask)
            shadow_draw.ellipse((padding + shadow_offset, padding + shadow_offset, 
                               size - padding + shadow_offset, size - padding + shadow_offset), fill=128)
            shadow_mask = shadow_mask.filter(ImageFilter.GaussianBlur(radius=shadow_blur))
            
            # Create shadow layer
            shadow_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            shadow_color = Image.new("RGBA", (size, size), (0, 0, 0, 80))  # Semi-transparent black
            shadow_img.paste(shadow_color, (0, 0))
            shadow_img.putalpha(shadow_mask)
            
            # Composite shadow with main image
            final_img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
            final_img = Image.alpha_composite(final_img, shadow_img)
            
            # Apply the circular mask to the resized image
            output_img.paste(img, (0, 0))
            output_img.putalpha(mask)
            
            # Composite the main image over the shadow
            final_img = Image.alpha_composite(final_img, output_img)
            
            # Save as PNG to preserve transparency
            final_img.save(output_path, "PNG")
            print(f"Processed image with drop shadow saved to {output_path}")
            
    except FileNotFoundError:
        print(f"Error: File '{input_path}' not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Usage: python image_processor.py <input_image_path> <output_image_path> [size] [padding]")
        print("Default size is 42 pixels, default padding is 2 pixels")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    size = int(sys.argv[3]) if len(sys.argv) >= 4 else 42
    padding = int(sys.argv[4]) if len(sys.argv) == 5 else 2
    
    process_image(input_path, output_path, size, padding)