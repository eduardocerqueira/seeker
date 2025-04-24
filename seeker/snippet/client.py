#date: 2025-04-24T16:48:29Z
#url: https://api.github.com/gists/91c97de01016af8e09ae96c4ae894756
#owner: https://api.github.com/users/abuibrahimjega

import requests
import argparse
import os
from PIL import Image
import io

def remove_text_from_image(image_path, api_url, languages='en', inpaint_radius=3):
    """
    Send an image to the API to remove text
    
    Args:
        image_path: Path to the input image
        api_url: URL of the API endpoint
        languages: Comma-separated list of language codes (e.g., 'en,fr')
        inpaint_radius: Radius for inpainting algorithm
    
    Returns:
        Path to the processed image
    """
    # Prepare the form data
    with open(image_path, 'rb') as f:
        files = {'image': (os.path.basename(image_path), f, 'image/jpeg')}
        data = {
            'languages': languages,
            'inpaint_radius': str(inpaint_radius)
        }
        
        # Send the request
        response = requests.post(api_url, files=files, data=data)
    
    # Check if request was successful
    if response.status_code == 200:
        # Get the output filename
        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)
        output_path = f"{name}_cleaned{ext}"
        
        # Save the response content as an image
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Text removed successfully! Saved as {output_path}")
        return output_path
    else:
        try:
            error_data = response.json()
            print(f"❌ Error: {error_data.get('detail', 'Unknown error')}")
        except:
            print(f"❌ Error: {response.text}")
        return None

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Remove text from an image using the API')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--api-url', type=str, default='http://localhost:5000/remove-text', 
                        help='URL of the API endpoint')
    parser.add_argument('--languages', type=str, default='en', 
                        help='Comma-separated list of language codes (e.g., "en,fr")')
    parser.add_argument('--inpaint-radius', type=int, default=3, 
                        help='Radius for inpainting algorithm')
    
    args = parser.parse_args()
    
    # Call the API
    output_path = remove_text_from_image(
        args.image_path, 
        args.api_url, 
        args.languages, 
        args.inpaint_radius
    )
    
    # Display the original and processed images
    if output_path:
        # Open original and processed images
        original = Image.open(args.image_path)
        processed = Image.open(output_path)
        
        # Display images (requires matplotlib)
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(original)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            axes[1].imshow(processed)
            axes[1].set_title('Text Removed')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib not available for displaying images.") 