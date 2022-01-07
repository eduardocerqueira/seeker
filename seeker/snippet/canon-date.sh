#date: 2022-01-07T17:11:31Z
#url: https://api.github.com/gists/2656d58cdc0e1ed850d21afab8c4f6e1
#owner: https://api.github.com/users/lennybe

###############################################################
# canon-date.sh
# Renames photos taken by Canon Digital Cameras into something
# more sortable and useful.
#
# Depends on exiftool 
#
# Download, make executable, and then run ./canon-date.sh in the 
# directory of the photos you want to rename.  The script will 
# create sub-directories by year and month.
###############################################################

exiftool '-FileName<CreateDate' -d %Y-%m/%Y%m%d_%H%M_%S_%f.%%e . 