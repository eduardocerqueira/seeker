#date: 2022-04-18T17:06:17Z
#url: https://api.github.com/gists/03d7fbe91bd0fdfdd10ef56ae461f639
#owner: https://api.github.com/users/taufiqibrahim

# Create directory for ftp
export NONROOTUSER=$USER
sudo mkdir -p /ftpdata
sudo chown -R $NONROOTUSER:$NONROOTUSER /ftpdata
mkdir -p /ftpdata/ftpuser/buckets/$BUCKET_NAME
mkdir -p /ftpdata/ftpuser/passwd
