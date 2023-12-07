#date: 2023-12-07T17:05:33Z
#url: https://api.github.com/gists/63a903a79858dc9df7688d57500c3f30
#owner: https://api.github.com/users/abstractml

import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Combine words into sentences
df_imdb['text'] = df_imdb['words'].apply(' '.join)

# Remove stopwords and perform TF-IDF transformation
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
X_imdb = tfidf_vectorizer.fit_transform(df_imdb['text'])
y_imdb = df_imdb['category']

# Split the data into training and testing sets
X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb = train_test_split(X_imdb, y_imdb, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
naive_bayes_imdb = MultinomialNB()
naive_bayes_imdb.fit(X_train_imdb, y_train_imdb)

# Make predictions
y_pred_imdb = naive_bayes_imdb.predict(X_test_imdb)

# Evaluate the model
accuracy_imdb = accuracy_score(y_test_imdb, y_pred_imdb)
conf_matrix_imdb = confusion_matrix(y_test_imdb, y_pred_imdb)
classification_rep_imdb = classification_report(y_test_imdb, y_pred_imdb)

print(f'Accuracy: {accuracy_imdb}')
print(f'Confusion Matrix:\n{conf_matrix_imdb}')
print(f'Classification Report:\n{classification_rep_imdb}')