#date: 2022-11-14T17:27:01Z
#url: https://api.github.com/gists/b75405c6f571f5285c17ec102a9ab538
#owner: https://api.github.com/users/gabri-al

import pandas as pd
import numpy as np

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
### ### ### Function to extract relevant fields from a playlist
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
def extract_from_playlist(playlist_id, genre):
    '''This function will return the following info from the provided playlist_id: artist, title, album, popularity.
    Genre is a string field which is a function input'''
    # Launch API
    try:
        results_ = sp.playlist(playlist_id)
    except:
        is_error = 1
        final_ = []
        print("Error occurred")
    else:
        is_error = 0

        # Get total tracks
        tot_ = results_['tracks']['total']
        
        # For each track, retrieve fields; results to be otained in batches of 100 records
        final_ = []
        
        for off_ in np.arange(0, tot_, 100):
            try:
                results_batch_ = sp.playlist_tracks(playlist_id = playlist_id, limit=100, offset=off_,
                                                    additional_types=('track', ))
            except:
                is_error = 1
            else:
                for i in np.arange(0, np.min([tot_, 100]), 1):
                    _artist = results_batch_['items'][i]['track']['artists'][0]['name']
                    _title = results_batch_['items'][i]['track']['name']
                    _album = results_batch_['items'][i]['track']['album']['name']
                    _popularity = results_batch_['items'][i]['track']['popularity']
                    final_.append([genre, _artist, _title, _album, _popularity])
    
    return is_error, final_

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
### ### ### Call API
### ### ### ### ### ### ### ### ### ### ### ### ### ### ### 
# Initialise an empty df
finaldf_ = pd.DataFrame({'Genre':[], 'Artist':[], 'Title':[], 'Album':[], 'Popularity':[]})

_metal_pl = '37i9dQZF1EQpgT26jgbgRI' # Metal mix https://open.spotify.com/playlist/37i9dQZF1EQpgT26jgbgRI
_pop_pl = '37i9dQZEVXbMDoHDwVN2tF' # Top Global 50 https://open.spotify.com/playlist/37i9dQZEVXbMDoHDwVN2tF

playlists = [_metal_pl, _pop_pl]
genres = ['Metal', 'Pop']

for j in np.arange(0, len(playlists), 1):
    p = playlists[j]
    g = genres[j]
    # Retrieve info
    err_, info_ = extract_from_playlist(p, g)
    # Append to df
    for i in info_:
        finaldf_.loc[len(finaldf_)] = i

print(finaldf_.head())