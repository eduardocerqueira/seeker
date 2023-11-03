#date: 2023-11-03T16:46:17Z
#url: https://api.github.com/gists/e128ca7f544af5fa545438d3764cf7cb
#owner: https://api.github.com/users/sayzame

from freeGPT import AsyncClient
from asyncio import run
import streamlit as st

async def main():
    st.title("💬 Chatbot")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Чем я могу помочь?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

        while True:
            try:
                resp = await AsyncClient.create_completion("gpt3", (f"пиши ответ на русском языке. {prompt}."))
                msg = resp
                st.session_state.messages.append({"role": "assistant", "content": msg})
                st.chat_message("assistant").write(msg)
                break
            except Exception as e:
                st.write(f"🤖: {e}")

if __name__ == "__main__":
    run(main())