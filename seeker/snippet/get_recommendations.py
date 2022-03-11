#date: 2022-03-11T16:53:59Z
#url: https://api.github.com/gists/28c44c37c88339a7e2b3a7baae8dd1b3
#owner: https://api.github.com/users/anarabiyev

def get_recommendations(title, cosine_sim=cosine_sim):
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    
    return df['title'].iloc[movie_indices]