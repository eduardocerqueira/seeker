#date: 2023-05-05T16:49:08Z
#url: https://api.github.com/gists/155d130d9825a52f343545f178070d0e
#owner: https://api.github.com/users/EnkrateiaLucca

from flask import Flask, render_template, request
import openai
import os

# Set up the API key
openai.api_key = "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"

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        topic = request.form['topic']
        system_profile = "You are an AI trained to provide helpful examples on various topics and concepts."
        prompt_question = f"Provide a variety of examples for the concept of {topic}."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_profile},
                {"role": "user", "content": prompt_question},
            ]
        )
        examples = response["choices"][0]["message"]["content"]
        return render_template('index.html', examples=examples)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)