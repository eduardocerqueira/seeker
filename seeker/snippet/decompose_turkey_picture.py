#date: 2023-11-23T17:08:41Z
#url: https://api.github.com/gists/f7884fe128b425755f52daec56cf7b62
#owner: https://api.github.com/users/ncalm

import numpy as np
import pandas as pd

def image_to_excel_color_codes(image_path, pixel_width):
    # Open the image
    img = Image.open(image_path)

    # Calculate the new height to maintain aspect ratio
    aspect_ratio = img.height / img.width
    new_height = int(pixel_width * aspect_ratio)

    # Resize the image while maintaining aspect ratio
    img_resized = img.resize((pixel_width, new_height))

    # Convert image to RGB (if it's not already in that format)
    img_rgb = img_resized.convert('RGB')

    # Create an array of RGB values
    rgb_array = np.array(img_rgb)

    # Convert the RGB values to hexadecimal
    hex_array = np.vectorize(lambda r, g, b: '#{:02x}{:02x}{:02x}'.format(r, g, b))(rgb_array[:,:,0], rgb_array[:,:,1], rgb_array[:,:,2])

    # Convert to a DataFrame
    df_hex = pd.DataFrame(hex_array)

    # Save the DataFrame to a CSV file
    csv_path = image_path.rsplit('.', 1)[0] + '_colors.csv'
    df_hex.to_csv(csv_path, index=False, header=False)
    
    return csv_path, img_resized

# Call the function with the provided image and desired pixel width
csv_file_path, processed_img = image_to_excel_color_codes('path/to/image', 50)

# Show the resized image and return the path to the CSV file
processed_img.show(), csv_file_path
