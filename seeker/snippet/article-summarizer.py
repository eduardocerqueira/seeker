#date: 2022-12-19T16:51:48Z
#url: https://api.github.com/gists/1ca1fee4712e9d59957c2e83da868bca
#owner: https://api.github.com/users/avrabyt

# Import the required libraries
import openai
# import os
import streamlit as st

# Set the GPT-3 API key
openai.api_key = "**********"

# Read the text of the article from a file
# with open("article.txt", "r") as f:
#     article_text = f.read()
article_text = st.text_area("Enter your scientific texts to summarize")
output_size = st.radio(label = "What kind of output do you want?", 
                    options= ["To-The-Point", "Concise", "Detailed"])

if output_size == "To-The-Point":
    out_token = "**********"
elif output_size == "Concise":
    out_token = "**********"
else:
    out_token = "**********"

if len(article_text)>100:
    if st.button("Generate Summary",type='primary'):
    # Use GPT-3 to generate a summary of the article
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt="Please summarize this scientific article for me in a few sentences: " + article_text,
            max_tokens = "**********"
            temperature = 0.5,
        )
        # Print the generated summary
        res = response["choices"][0]["text"]
        st.success(res)
        st.download_button('Download result', res)
else:
    st.warning("Not enough words to summarize!")