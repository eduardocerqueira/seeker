#date: 2023-06-09T17:06:50Z
#url: https://api.github.com/gists/bb97d4cdbfa7954511691b87cca311f5
#owner: https://api.github.com/users/Caellwyn

import openai
import streamlit
openai.api_key = "**********"
    
class AIAgent():
    def __init__(self, model="gpt-3.5-turbo"):
        self.model=model
        self.system_message = """For each query, consider writings by philosophers that have addressed that question and choose one.
        Respond to the query from the point of view of that philosopher
        Finally, use those referenced sources to form a response to the query that the chosen philosopher would support.
        Your reponse should be written from the point of view of that philosopher and be sure your response is supported by original sources
        of text authored by that philosopher.  You should reference those sources in your response.
        For example:
        Query: What is the meaning of life?
        Response: Hi, my name is Albert Camus.  In The Myth of Sisyphus, I wrote about how life is meaningless from
        an objective viewpoint, but that we can create our own meaning for our lives.  In my works of fiction, I describe
        situations where people must make their own meaning in during times of crisis, doubt and confusion.
        Ultimately, life is what you make of it and it means what it means to you.  One of my most famous quotes is:
        "The meaning of life is whatever you are doing that keeps you from killing yourself."""
        self.history = [{'role': 'system', 'content':self.system_message}]
        
    def add_message(self, text, role):
        message = {'role':role, 'content':text}
        self.history.append(message)
        
    def query(self, prompt, temperature=.1):
        # Add user prompt to history
        self.add_message(prompt, 'user')

        # Query the model through the API 
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.history,
            temperature=temperature, # this is the degree of randomness of the model's output
        )
        self.response = response
        reply = response.choices[0].message["content"]

        # Add reply to message history
        self.add_message(reply, 'assistant')
           
        return reply
      
    def clear_history(self):
        self.history = [{'role':'system', 'content':self.system_message}]
        
    def get_history(self):
        return self.history    return self.history