#date: 2021-09-27T16:53:47Z
#url: https://api.github.com/gists/e37c3692578e4eb93234e97ecadeb049
#owner: https://api.github.com/users/naviji

#!/usr/bin/env sh

# first check to see if mongo service is running. you can't delete any files until the service stops so perform a quick check.
launchctl list | grep mongo
  # NOTE: the pipe | symbol means the commands on the right performs on the output from the left
  # grep is a string search utility. `grep mongo` means search for the substring mongo
 
# use the unload command to end the mongo service. this is required to 'unlock' before removing the service.
#    first look for the file to delete
MONGO_SERVICE_FILE=$(ls ~/Library/LaunchAgents/*mongodb*)
#    the following produces something like launchctl unload ~/Library/LaunchAgents/homebrew.mxcl.mongodb.plist
launchctl unload $MONGO_SERVICE_FILE


  # NOTE: if there is an error with the unload command it means it's not recognizing the correct filename
  # you can try searching for the correct filename within the ~/Library/LaunchAgents/ folder. see line 39 for the command.
  
  
  # NOTE: another mention about the note above to ensure reader sees the message above about correct filename.
 

# then you can remove the service. the name of the service after the word remove should match the output from running the command shown in line 4
# basically you're filling in the pattern launchctl remove <service-name>
launchctl remove homebrew.mxcl.mongodb

# kill the process
pkill -f mongod

# remove the mongodb service file. again, please make sure the filename is correct. 
#   this command removes the plist file that was found above. It should create a command that works to the effect of
#       rm -f ~/Library/LaunchAgents/homebrew.mxcl.mongodb.plist
rm -f $MONGO_SERVICE_FILE

# remove data directories
rm -rf /usr/local/var/mongod
  # NOTE: the flag options here include both r and f 
  # using -rf together means you're telling the machine to 'delete all items in the specified path subtree without confirmation'

# uninstal mongodb using brew
brew uninstall mongodb

# double check existence of mongodb in both folders
ls -al /usr/local/bin/*mongo*
ls -al ~/Library/LaunchAgents/*mongo*
  # NOTE: The asterisk indicate wildcard using regular expression (regex) notation along with substring `mongo`
  # This regex syntax has same purpose as using grep when searching for the existence of filename(s)
  # so instead of soley using the `ls` command you can also combine with grep `ls -al ~/Library/LaunchAgents | grep mongo`
  
  
# now you should have it entirely removed and can reinstall mongodb or whatever your heart desires