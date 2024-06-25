#date: 2024-06-25T16:39:19Z
#url: https://api.github.com/gists/79967f9afabd312449bbeb9918f69939
#owner: https://api.github.com/users/L-narendar-kumar

# views.py
from django.shortcuts import render
from .models import Movie
from difflib import get_close_matches # For handling potential typos in movie titles
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import difflib

def get_movie_recommendations(movie_title):
    movies_data = pd.read_csv('new_movies_data.csv')
    movies_data['title'] = movies_data['title'].fillna('Unknown')
    combined_features = movies_data['title']
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    list_of_all_titles = movies_data['title'].tolist()
    
    find_close_match = difflib.get_close_matches(movie_title,list_of_all_titles)
    close_match = find_close_match[0]
    index_of_movie = movies_data[movies_data.genres == close_match].index[0]
    similarity_score = list(enumerate(similarity[index_of_movie]))

    sorted_similar_movies = sorted(similarity_score,key = lambda x:x[1],reverse = True)
    
    result = []
    for i, j in enumerate(sorted_similar_movies[:5]): 
        movie_id = movies_data.iloc[j[0]].movie_id
        title_from_index = movies_data.iloc[j[0]].title
        poster_url = movies_data.iloc[j[0]].poster_url

        # Fetch the image and encode it as base64
        

        result.append((poster_url, title_from_index))

    return result 

def get_genre_recommendations(genre, num_recommendations=5):
    movies_data = pd.read_csv('new_movies_data.csv')
    movies_data['genres'] = movies_data['genres'].fillna('Unknown')
    combined_features = movies_data['genres']
    vectorizer = TfidfVectorizer()
    feature_vectors = vectorizer.fit_transform(combined_features)
    similarity = cosine_similarity(feature_vectors)
    list_of_all_genres = movies_data['genres'].tolist()
    
    find_close_match = difflib.get_close_matches(genre,list_of_all_genres)
    close_match = find_close_match[0]
    index_of_movie = movies_data[movies_data.genres == close_match].index[0]
    similarity_score = list(enumerate(similarity[index_of_movie]))

    sorted_similar_movies = sorted(similarity_score,key = lambda x:x[1],reverse = True)
    
    result = []
    for i, j in enumerate(sorted_similar_movies[:5]): 
        movie_id = movies_data.iloc[j[0]].movie_id
        title_from_index = movies_data.iloc[j[0]].title
        poster_url = movies_data.iloc[j[0]].poster_url

        # Fetch the image and encode it as base64
        

        result.append((poster_url, title_from_index))

    return result

def movie_recommendations(request):
    if request.method == 'POST':
        search_type = request.POST.get('search_type') 
        if search_type == 'title':
            search_title = request.POST.get('movie_title')
            # Handle potential typos in the movie title (same as before)
            all_titles = Movie.objects.values_list('title', flat=True)
            close_matches = get_close_matches(search_title, all_titles, n=1, cutoff=0.6)
            if close_matches:
                search_title = close_matches[0]
            recommendations = get_movie_recommendations(search_title)
        elif search_type == 'genre':
            search_genre = request.POST.get('genre')
            recommendations = get_genre_recommendations(search_genre)
        else:
            recommendations = []  # Handle invalid search type
        
        return render(request, 'movie_recommendations.html', {'recommendations': recommendations})
    else:
        return render(request, 'movie_search.html')