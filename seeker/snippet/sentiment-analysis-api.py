#date: 2023-08-22T16:47:53Z
#url: https://api.github.com/gists/a7a5cd026c2c45d6469740624d95b98f
#owner: https://api.github.com/users/hiddev-id

# %%

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# %%
pretrained = "mdhugol/indonesia-bert-sentiment-classification"

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.info(f"loading model from {pretrained}...")

model = AutoModelForSequenceClassification.from_pretrained(pretrained)
tokenizer = "**********"

sentiment_analysis = "**********"=model, tokenizer=tokenizer)

label_index = {'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative'}

# %%
# create api to get the sentiment of a new sentence (input from user)
# set port to 3300

# add rate limit to the api (limit to 10 request per minute)

from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
from waitress import serve

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    storage_uri="memory://",
)

@app.after_request
def after_request(response):
    # log the endpoint and the request body
    logging.info(f"endpoint: {request.path}")
    logging.info(f"request body: {request.get_json()}")
    return response

@app.route('/sentiment', methods=['POST'])
@limiter.limit("10 per minute", error_message="Jangan keseringan dong, coba lagi nanti yaa")
def sentiment():
    data = request.get_json()

    # make sure data['text'] is not empty and is a string and more than 5 characters, and maximum 200 characters
    if not data or 'text' not in data or not isinstance(data['text'], str) or len(data['text']) < 5 or len(data['text']) > 200:
      return jsonify({
          'error': 'text is required and must be a string, more than 5 characters and maximum 200 characters'
      })

    result = sentiment_analysis(data['text'])
    return jsonify({
        'label': label_index[result[0]['label']],
        'score': result[0]['score']
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3300))
    is_prod = os.environ.get('APP_ENV', 'development') == 'production'

    if is_prod:
      serve(app, host="0.0.0.0", port=port)
    else:
      app.run(debug=True, host='0.0.0.0', port=port)
      app.run(debug=True, host='0.0.0.0', port=port)