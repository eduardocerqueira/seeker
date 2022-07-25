#date: 2022-07-25T16:59:03Z
#url: https://api.github.com/gists/406b1003202e20a298df6f5780812014
#owner: https://api.github.com/users/Abhayparashar31

'''
Run Using `python app.py`
'''
import pickle
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle

import gradio as gr


## Loading model and cv
cv = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('review.pkl','rb'))

## User Input
review = gr.Textbox(label = "Enter Your Review...")

## Cleaning Function
def clean_review(review):
    new_review = re.sub('[^a-zA-Z]', ' ', review)
    new_review = new_review.lower()
    new_review = new_review.split()   

    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    new_review = [ps.stem(word) for word in new_review if not word in set(all_stopwords)]

    new_review = ' '.join(new_review)
    new_corpus = [new_review]
    new_X_test = cv.transform(new_corpus).toarray()
    return new_X_test

## Main Function
def make_prediction(review):

    new_X_test = clean_review(review)

    ### Prediction
    pred = model.predict(new_X_test)
    if review!="":
        if pred==1:
            return "Positive 😀"
        else:
            return "Negative 😑"

app =  gr.Interface(fn = make_prediction, inputs=review, outputs="text")
app.launch(share=True) ## If True, Generates Public Share Link That Is Valued For 72 Hours. 