#date: 2022-03-03T17:13:07Z
#url: https://api.github.com/gists/074ef76477939d618f86ee5364f62ecf
#owner: https://api.github.com/users/00100000

#!/bin/sh
homeDir=~
backupDir=~/Documents/Github/configs
toBackup=( "backup.sh" ".vimrc" ".vim" ".git-credentials" ".gitconfig" "Library/Preferences/com.googlecode.iterm2.plist" )

if [ $1 == "push" ]
then
	for i in "${toBackup[@]}"
	do
	:
		cp -R $homeDir/"$i" $backupDir
	done
fi

if [ $1 == "pull" ]
then
	for i in "${toBackup[@]}"
	do
	:
		cp -R $backupDir/"$i" $homeDir
	done
fi