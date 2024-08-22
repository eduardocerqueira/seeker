#date: 2024-08-22T16:48:55Z
#url: https://api.github.com/gists/0e4c7d9db09ead5df0c10b068d5b843f
#owner: https://api.github.com/users/quicksocialvaud

import openai
import gradio as gr

client = openai.OpenAI()
threads = {}

def predict(message, history, request: gr.Request):
    global threads
    if request.session_hash in threads:
        thread = threads[request.session_hash]
    else:
        thread = client.beta.threads.create()
        threads[request.session_hash] = thread
        
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=message)
    

    message = ""
    with client.beta.threads.runs.stream(
    thread_id=thread.id,
    assistant_id="<assistant-id>"
    ) as stream:
        for event in stream:
            # Print the text from text delta events
            if event.event == "thread.message.delta" and event.data.delta.content:
                message += event.data.delta.content[0].text.value
                yield message
    
gr.ChatInterface(predict).launch(show_error=True)