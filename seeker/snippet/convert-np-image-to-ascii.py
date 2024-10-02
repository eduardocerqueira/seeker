#date: 2024-10-02T16:56:22Z
#url: https://api.github.com/gists/9f96a8b039653094c9a69d87a021df33
#owner: https://api.github.com/users/Joel-hanson

def image_to_ascii(img, width=100, height=50):
    # Convert to numpy array if it's not already
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    # Ensure the image is in the correct format (H, W, C)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    
    # Normalize the image if it's not in [0, 1] range
    if img.max() > 1.0:
        img = img / 255.0
    
    # Resize the image
    img_resized = np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((width, height)))
    
    # Convert to grayscale
    if img_resized.ndim == 3:
        img_gray = np.mean(img_resized, axis=2)
    else:
        img_gray = img_resized
    
    # Normalize grayscale to [0, 1]
    img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min())
    
    # Define the ASCII characters to use (from darkest to lightest)
    ascii_chars = ['@', '#', 'S', '%', '?', '*', '+', ';', ':', ',', '.']
    
    # Convert grayscale values to ASCII characters
    ascii_img = []
    for row in img_gray:
        ascii_row = [ascii_chars[min(int(pixel * len(ascii_chars)), len(ascii_chars) - 1)] for pixel in row]
        ascii_img.append(''.join(ascii_row))
    
    print('\n'.join(ascii_img))