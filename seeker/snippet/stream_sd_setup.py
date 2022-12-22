#date: 2022-12-22T16:42:17Z
#url: https://api.github.com/gists/5be76e3f0144593ef4913cc393813166
#owner: https://api.github.com/users/StatsGary

 SAVE_LOCATION = 'prompt.jpg'
 # Create the page title 
 st.set_page_config(page_title='Diffusion Model generator', page_icon='fig/favicon.ico')
 # Create page layout
 with open('style.css') as f:
     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
 st.title('Image generator using Stable Diffusion')
 st.caption('An app to generate images based on text prompts with a :blue[_Stable Diffusion_] model :sunglasses:')