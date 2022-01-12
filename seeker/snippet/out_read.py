#date: 2022-01-12T17:02:29Z
#url: https://api.github.com/gists/527bb05f187c0e89246d62ed4fa31546
#owner: https://api.github.com/users/haykaza

# read embeddings from .out file exported from the Labs UI 
X = pd.read_json('BERT-RNA/data_phrasebook.out')
y = input_df['label']