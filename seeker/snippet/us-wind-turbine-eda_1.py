#date: 2022-09-02T17:14:47Z
#url: https://api.github.com/gists/c81479f5b020e2bef0c528827a1e29a8
#owner: https://api.github.com/users/yousafmufc

from google.colab import drive
drive.mount('/content/drive')

#unzipping the file to access the CSV dataset
!unzip "/content/drive/My Drive/Jovian/US_Wind_DB.zip"

data_filename = '/content/uswtdb_v5_1_20220729.csv'