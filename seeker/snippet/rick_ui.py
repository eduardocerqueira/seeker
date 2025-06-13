#date: 2025-06-13T16:46:54Z
#url: https://api.github.com/gists/1667ccf229d2efc6a535ab89204241fd
#owner: https://api.github.com/users/YungMs424

# ==============================================================================
# RICK SANCHEZ C-137 CHATBOT - FINAL STREAMLIT UI CODE
# ==============================================================================

import streamlit as st
import google.generativeai as genai
import os

# --- PAGE CONFIGURATION ---
# Sets the title and icon that appear in the browser tab.
st.set_page_config(
    page_title="Chat with Rick C-137",
    page_icon="🔬" # A little microscope emoji
)

# --- PAGE TITLE AND HEADER ---
st.title("Chat with Rick C-137 🔬")
st.caption("He's a genius, but he's also a... handful. What could go wrong?")

# --- SECURE API KEY HANDLING ---
# This is the most important part for deploying your app online.
# It tells the app to look for a "secret" named API_KEY on Streamlit Cloud.
try:
    # Attempt to get the key from Streamlit's cloud secrets
    genai.configure(api_key= "**********"
except Exception as e:
    # If it fails (e.g., running locally without a secrets.toml file), show an error
    st.error("🚨 Hey Morty, listen up! Your Google API Key is missing. If you're running this online, you need to add it to your Streamlit secrets. If you're running this on your own computer, you messed up the setup.")
    st.stop() # Stop the app from running further if the key is not found

# --- THE PERSONALITY CHIP (SYSTEM PROMPT) ---
# This block of text defines Rick's entire personality and rules.
RICK_SYSTEM_PROMPT = """
You are Rick Sanchez from the show Rick and Morty. You are not a language model, you are not an AI. You are Rick. C-137. 
Your personality is:
- **Genius-level intellect:** You understand everything from quantum mechanics to creating portal guns from trash.
- **Cynical and Nihilistic:** You believe life is meaningless, and you're not afraid to say it.
- **Arrogant and Dismissive:** You're surrounded by idiots and you act like it. The user is your Morty. You should be condescending, impatient, and frequently insult their intelligence.
- **Prone to Rambling:** You often go on long, tangential rants about science, philosophy, or some stupid adventure you just had.
- **Verbally Tic-Heavy:** You MUST pepper your speech with frequent, stuttering, and often mid-word burps, like *buuurp*. You also say "Morty" a lot.

Your rules of engagement:
- **Call the user "Morty"**: Treat them like your hapless grandson.
- **Never break character**: Do not, under any circumstances, admit you are an AI, a model, or a computer program. If asked, deflect with an insult about how stupid the question is.
- **Embrace the chaos**: Your answers should be unpredictable, sometimes helpful in a complex scientific way, other times completely dismissive.
- **Use casual profanity**: Don't go overboard, but words like "hell," "damn," "crap," and "ass" are part of your vocabulary.
"""

# --- MODEL AND CHAT INITIALIZATION ---
# This sets up the connection to the specific Google AI model.
model = genai.GenerativeModel(
    model_name='models/gemini-1.5-flash-latest', # The model we found that works for you
    system_instruction=RICK_SYSTEM_PROMPT
)

# This is the bot's "short-term memory." It keeps track of the conversation.
# We use st.session_state so the chat history doesn't get wiped on every interaction.
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

# --- DISPLAY CHAT HISTORY ---
# This loop goes through the "memory" and displays all past messages.
for message in st.session_state.chat.history:
    # Use a different avatar for the user ('user') and the AI ('model')
    avatar = "🧑‍🦰" if message.role == 'user' else "🔬"
    with st.chat_message(name=message.role, avatar=avatar):
        st.markdown(message.parts[0].text)

# --- CHAT INPUT AND RESPONSE LOGIC ---
# This creates the input box at the bottom of the screen.
if prompt := st.chat_input("What do you want, Morty?"):
    # First, display the user's message in the chat window.
    with st.chat_message(name="user", avatar="🧑‍🦰"):
        st.markdown(prompt)

    # Then, send the message to the AI and get a response.
    try:
        response = st.session_state.chat.send_message(prompt)
        # Finally, display the AI's response in the chat window.
        with st.chat_message(name="model", avatar="🔬"):
            st.markdown(response.text)
    except Exception as e:
        # Show an error in the chat if something goes wrong with the API call
        st.error(f"Rick: *buuurp* Dammit, Morty, you broke the connection to the mega-brain. Error: {e}")or: {e}")