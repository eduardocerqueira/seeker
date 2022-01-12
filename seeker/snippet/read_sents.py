#date: 2022-01-12T17:07:35Z
#url: https://api.github.com/gists/1b3ff1d2793c4f12036a515e6698159a
#owner: https://api.github.com/users/haykaza

# read embeddings from Labs UI exported .out file
sentiment1 = pd.read_json('BERT-RNA/sentiment1.out')
sentiment2 = pd.read_json('BERT-RNA/sentiment2.out')
sentiment3 = pd.read_json('BERT-RNA/sentiment3.out')
sentiment4 = pd.read_json('BERT-RNA/sentiment4.out')

#merge sentiment headlines into one dataframe
sentiment = pd.concat([sentiment1, sentiment2, sentiment3, sentiment4], ignore_index = True, axis = 0)
sentiment