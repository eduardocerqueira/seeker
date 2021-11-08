#date: 2021-11-08T16:58:49Z
#url: https://api.github.com/gists/34c06cb82c2a70559775203e6cc3f479
#owner: https://api.github.com/users/irenechang1510

# get happy playlist
happy_playlist = [
    'spotify:playlist:37i9dQZF1DX3rxVfibe1L0', 
    'spotify:playlist:37i9dQZF1DX6GwdWRQMQpq', 
    'spotify:playlist:37i9dQZF1DX66m4icL86Ru', 
    'spotify:playlist:0RH319xCjeU8VyTSqCF6M4',
    'spotify:playlist:37i9dQZF1DWZKuerrwoAGz']
happy = pd.DataFrame(columns=['name', 'artist', 'track_URI', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'playlist'])
happy = get_all_songs(happy, username, happy_playlist)

ones = np.ones(len(happy)) # create the label for 'happy' class (1)
happy.insert(3, 'label', ones)

# get sad playlist
sad_playlist = [
    'spotify:playlist:37i9dQZF1DX3YSRoSdA634',
    'spotify:playlist:37i9dQZF1DWSqBruwoIXkA',
    'spotify:playlist:4yXfnhz0BReoVfwwYRtPBm'
]
sad = pd.DataFrame(columns=['name', 'artist', 'track_URI', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence', 'playlist'])
sad = get_all_songs(sad, username, sad_playlist)

zeros = np.zeros(len(sad))
sad.insert(3, 'label', zeros) # create the label for 'sad' class (0)


#merge them 
full_data = happy.merge(sad, how='outer')