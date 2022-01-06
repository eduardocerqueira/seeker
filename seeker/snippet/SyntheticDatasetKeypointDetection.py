#date: 2022-01-06T17:08:00Z
#url: https://api.github.com/gists/1d30705ab34e1d416febece09fa6db60
#owner: https://api.github.com/users/alexppppp

# Let's look at a random object and its binary mask

img_path = files_imgs[9]
mask_path = files_masks[9]

img, mask = get_img_and_mask(img_path, mask_path)

print("Image file:", img_path)
print("Mask file:", mask_path)
print("\nShape of the image of the object:", img.shape)
print("Shape of the binary mask:", mask.shape)

fig, ax = plt.subplots(1, 2, figsize=(16, 7))
ax[0].imshow(img)
ax[0].set_title('Object', fontsize=18)
ax[1].imshow(mask)
ax[1].set_title('Binary mask', fontsize=18);