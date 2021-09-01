#date: 2021-09-01T13:09:50Z
#url: https://api.github.com/gists/53f25da6c5010b2d4241c7eb4f4c5f4f
#owner: https://api.github.com/users/Ashcom-git

topic_keywords = []
for topic_weights in lda.components_:
  top_keyword_locs = (-topic_weights).argsort()[:30]
  topic_keywords.append(np.asarray(vectorizer.get_feature_names()).take(top_keyword_locs).tolist())

for i in range(len(topic_keywords)):
  print('Aspect {}'.format(i))
  print(' | '.join(topic_keywords[i]))