#date: 2021-11-10T17:01:37Z
#url: https://api.github.com/gists/8c433e0220ad91a43291d8dd64294ff6
#owner: https://api.github.com/users/nagi1995

def datapipeline(text):
    text = decontracted(text)
    x_test = vectorizer.transform(text)
    prediction = model.predict(x_test)
    if prediction[0][1] >= .5:
        print("SARCASTIC")
    else:
        print("NOT SARCASTIC")