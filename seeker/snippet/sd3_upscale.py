#date: 2024-04-17T16:49:29Z
#url: https://api.github.com/gists/4927cdd6b67576db9a5c4ed6fcc1e43f
#owner: https://api.github.com/users/cavit99

import requests
import uuid
import time

api_key = "YOUR-STABILITY-API"

num_images = 2  # Specify the number of images to generate
prompt = "Here is your positive prompt"
negative_prompt = "here is your negative prompt" # Optional
seed = 0 # Set to zero for random
aspect_ratio = "2:3" #16:9 1:1 21:9 2:3 3:2 4:5 5:4 9:16 9:21
upscale_images = True  # Set to False to skip upscaling

unique_id = str(uuid.uuid4())  # Generate a unique UUID for the batch

# Generate all images first
generated_images = []
for i in range(num_images):
    response = requests.post(
        f"https://api.stability.ai/v2beta/stable-image/generate/sd3",
        headers={
            "authorization": f"Bearer {api_key}",
            "accept": "image/*"
        },
        files={"none": ''},
        data={
            "seed": seed,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "aspect_ratio": aspect_ratio
        },
    )

    if response.status_code == 200:
        filename = f"./{unique_id}_{i+1:02d}.png"  # Use the same UUID for the batch
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Image {i+1} generated and saved as {filename}")
        generated_images.append(filename)
    else:
        raise Exception(str(response.json()))

# Queue up upscale requests
upscale_requests = []
if upscale_images:
    for i, filename in enumerate(generated_images):
        upscale_response = requests.post(
            f"https://api.stability.ai/v2beta/stable-image/upscale/creative",
            headers={
                "authorization": f"Bearer {api_key}",
                "accept": "image/png"  # Request PNG format for the upscaled image
            },
            files={
                "image": open(filename, "rb")
            },
            data={
                "prompt": prompt
            },
        )
        
        if upscale_response.status_code == 200:
            generation_id = upscale_response.json().get('id')
            print(f"Upscaling request for Image {i+1} submitted. Generation ID: {generation_id}")
            upscale_requests.append((i, generation_id))
        else:
            raise Exception(str(upscale_response.json()))

# Fetch upscaled images
for i, generation_id in upscale_requests:
    while True:
        result_response = requests.get(
            f"https://api.stability.ai/v2beta/stable-image/upscale/creative/result/{generation_id}",
            headers={
                'accept': "image/*", 
                'authorization': f"Bearer {api_key}"
            },
        )
        
        if result_response.status_code == 202:
            # The upscaling request is still in progress
            print(f"Upscaling for Image {i+1} in progress. Waiting for 15 seconds...")
            time.sleep(15)  # Wait for 15 seconds before checking again
        elif result_response.status_code == 200:
            print(f"Upscaling for Image {i+1} complete!")
            upscaled_filename = f"./{unique_id}_{i+1:02d}_upscaled.png"  
            with open(upscaled_filename, 'wb') as file:
                file.write(result_response.content)
            print(f"Upscaled Image {i+1} saved as {upscaled_filename}")
            break  # Exit the loop and move to the next upscale request
        else:
            raise Exception(str(result_response.json()))