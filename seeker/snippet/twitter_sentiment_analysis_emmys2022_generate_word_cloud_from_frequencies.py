#date: 2022-10-12T17:08:47Z
#url: https://api.github.com/gists/7679e7a49cc2e641fa3131498871c429
#owner: https://api.github.com/users/yavuzKomecoglu

from wordcloud import WordCloud

def generate_word_cloud_from_frequencies(freq_dict, status):

    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color = 'black',
        stopwords = stop_words).generate_from_frequencies(frequencies=freq_dict)
    
    plt.figure( figsize=(20,10), facecolor='k')
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

    wordcloud.to_file(SAVE_WORD_CLOUD_FILEPATH.format(status))