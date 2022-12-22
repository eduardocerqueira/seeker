#date: 2022-12-22T17:02:11Z
#url: https://api.github.com/gists/e5e67626e25c03d7b5459abedc867581
#owner: https://api.github.com/users/StatsGary

if len(prompt) > 0:
  st.markdown(f"""
  This will show an image using **stable diffusion** of the desired {prompt} entered:
  """)
  print(prompt)
  # Create a spinner to show the image is being generated
  with st.spinner('Generating image based on prompt'):
    sd = StableDiffusionLoader(prompt)
    sd.generate_image_from_prompt(save_location=SAVE_LOCATION)
    st.success('Generated stable diffusion model')

# Open and display the image on the site
image = Image.open(SAVE_LOCATION)
st.image(image)  