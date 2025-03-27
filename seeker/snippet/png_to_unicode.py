#date: 2025-03-27T17:04:49Z
#url: https://api.github.com/gists/661c1e38e080d7665a57f9ccb7563df9
#owner: https://api.github.com/users/baconismycopilot

import base64
import io
from PIL import Image

# Unicode characters from darkest to lightest
UNICODE_BLOCKS = "█▓▒░  "

def base64_to_image(base64_string):
    """Decode Base64 string to a PIL image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))

def image_to_unicode(image, width=80):
    """Convert image to a Unicode block character representation."""
    # Resize while maintaining aspect ratio
    aspect_ratio = image.height / image.width
    new_height = int(width * aspect_ratio * 0.5)  # Adjust for terminal character aspect ratio
    image = image.resize((width, new_height))

    # Convert to grayscale
    image = image.convert("L")

    # Map pixels to Unicode block characters
    pixels = image.getdata()
    unicode_str = "".join(UNICODE_BLOCKS[pixel // 51] for pixel in pixels)  # 256 levels mapped to 5 chars

    # Format as lines
    unicode_lines = [unicode_str[i:i+width] for i in range(0, len(unicode_str), width)]
    return "\n".join(unicode_lines)

# Example usage with a Base64 string (Replace with your own)
base64_string = "iVBORw0KGgoAAAANSUhEUgAA..."  # Replace with a valid Base64 string
image = base64_to_image(base64_string)
unicode_art = image_to_unicode(image, width=80)

print(unicode_art)
