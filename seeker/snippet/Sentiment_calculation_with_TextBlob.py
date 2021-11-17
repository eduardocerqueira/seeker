#date: 2021-11-17T17:03:31Z
#url: https://api.github.com/gists/aa680c268aa7989950f46bec9484cf9e
#owner: https://api.github.com/users/gdatavalley

#TextBlob single sentence polarity score calculation

#Importing libraries
from textblob import TextBlob

#Defining sentence to apply function on
blob = TextBlob('Today is such a great day')

#Printing out on the console the polarity score
blob.sentiment.polarity