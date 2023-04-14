#date: 2023-04-14T16:53:56Z
#url: https://api.github.com/gists/4c875120122dab0ee8fc56432bf21781
#owner: https://api.github.com/users/Adi-0987

from flask import Flask, request, jsonify, render_template
import mysql.connector
import openai

# Set up Flask app
app = Flask(__name__)

# Set up OpenAI API key
openai.api_key = "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"

# Set up MySQL database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password= "**********"
    database="project"
)


# Define endpoint to render index.html
@app.route("/")
def index():
    return render_template("index.html")

# Define endpoint for ChatGPT query
@app.route("/chat", methods=["POST"])
def chat():
    # Get query text from user input
    query = request.form.get("query")

    # Use OpenAI to generate response to user input
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=query,
        max_tokens= "**********"
        n=1,
        stop=None,
        temperature=0.7,
    )

    # Extract response text from OpenAI API response
    response_text = response["choices"][0]["text"]

    # Use response text to query MySQL database
    cursor = db.cursor()
    cursor.execute(response_text)
    result = cursor.fetchall()
    return result

# Define function to convert SQL query result into natural language using OpenAI API
def generate_result_text(result):
    prompt = "Convert the following SQL query result into natural language: " + str(result)
    completions = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens= "**********"
        n=1,
        stop=None,
        temperature=0.5,
    )
    return completions.choices[0].text.strip()

# Start Flask app
if __name__ == "__main__":
    app.run()
