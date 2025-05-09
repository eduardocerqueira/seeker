#date: 2025-05-09T17:08:40Z
#url: https://api.github.com/gists/2f293f3f3231bf2dfdec9899105d9b63
#owner: https://api.github.com/users/Deman3D

# %%
import replicate
model = replicate.models.get("prompthero/openjourney")
version = model.versions.get("9936c2001faa2194a261c01381f90e65261879985476014a0a37a334593a05eb")
PROMPT = "mdjrny-v4 style 360 degree equirectangular panorama photograph, Alps, giant mountains, meadows, rivers, rolling hills, trending on artstation, cinematic composition, beautiful lighting, hyper detailed, 8 k, photo, photography"
output = version.predict(prompt=PROMPT, width=1024, height=512)

# %%
# download the iamge from the url at output[0]
import requests
image = requests.get(output[0]).content


#save image as a file
tempfile = open("output.jpg", "wb")
tempfile.write(image)
tempfile.close()



# %%
# create a mask image of ize 1024 x 512 that is all black
mask_img = Image.new('L', (1024, 512), color = 'white')
# add white borders of width 128 on either side
mask_img.paste(0, (0, 0, 128, 512))
mask_img.paste(0, (896, 0, 1024, 512))
mask_img.save('mask.jpg')



# %%

# do seconds model
from PIL import Image
model = replicate.models.get("andreasjansson/stable-diffusion-inpainting")
version = model.versions.get("8eb2da8345bee796efcd925573f077e36ed5fb4ea3ba240ef70c23cf33f0d848")

PROMPT = PROMPT
# open output.jpg

file_object = open("output.jpg", "rb")
# generate image from file_object
input_image = Image.open(file_object)
# save left_most 128 pixels to temp image
temp_image = input_image.crop((0, 0, 128, 512))

# for input_image, put right-most 128 pixels to the left side
input_image.paste(input_image.crop((896, 0, 1024, 512)), (0, 0, 128, 512))

# put temp_image to the right side
input_image.paste(temp_image, (896, 0, 1024, 512))

# save the image
input_image.save('input.jpg')




# %%

input_object = open("input.jpg", "rb")
mask_object = open("mask.jpg", "rb")


output = version.predict(prompt=PROMPT, width=1024, height=512, image=input_object, mask=mask_object)
print(output)

image = requests.get(output[0]).content
tempfile = open("final_output.jpg", "wb")
tempfile.write(image)
tempfile.close()


# %%
# merge output.jpg and final_output.jpg

output_image = Image.open("output.jpg")
final_output_image = Image.open("final_output.jpg")

combined_image = Image.new('RGB', (2048-256, 512))
combined_image.paste(output_image, (0, 0))
combined_image.paste(final_output_image, (896, 0))
combined_image.save('combined.jpg')



# %%
# show combined image
combined_image.show()
