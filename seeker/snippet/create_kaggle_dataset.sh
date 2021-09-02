#date: 2021-09-02T16:58:51Z
#url: https://api.github.com/gists/19276244f44786698ee6a024da7c6cc9
#owner: https://api.github.com/users/arch-raven

# https://github.com/Kaggle/kaggle-api#api-credentials
pip install kaggle
echo "{'username':'INSERT_KAGGLE_USERNAME','key':'KAGGLE_KEY'}" > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# https://github.com/Kaggle/kaggle-api#create-a-new-dataset-version
# Put all files to upload inside a folder (example: spotify-millions)
# this command creates a `dataset-metadata.json` files, edit the specifiwd fields in this file
kaggle datasets init -p spotify-millions/
# running the following command uploads the folder to kaggle datasets
kaggle datasets create -p spotify-millions/ --public