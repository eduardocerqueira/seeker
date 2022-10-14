#date: 2022-10-14T17:24:07Z
#url: https://api.github.com/gists/78061a8f15451ca59fc6702efc3ddead
#owner: https://api.github.com/users/florent-brosse

%sh
kaggle competitions download -q -c airbus-ship-detection -p /local_disk0/
unzip -q /local_disk0/airbus-ship-detection.zip -d /local_disk0/airbus-ship-detection/