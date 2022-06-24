#date: 2022-06-24T17:05:20Z
#url: https://api.github.com/gists/7cce0d5040d9c2551efbcc99e21b88c6
#owner: https://api.github.com/users/syedali3762

#!/bin/bash
not_sudo=101
error=202
green='\033[0;32m'
white='\033[0m'
red='\033[0;31m'
if [[ $UID -eq "0" ]]
then
which nginx > /dev/null ||{
 echo -e "${red}Something went wrong, NGINX cannot be activated ${white}"
 exit $error
} && {
status=$(systemctl status nginx | grep Active | awk '{print $2}')
if [[ $status = "inactive" ]]
then
        echo -e  "${red}NGINX is Dead. Do you want to run NGINX [y/n]? ${white>
read input

if [[ $input = "y" ]]
then
       systemctl start nginx
else
        echo -e "${red}Something went wrong, NGINX cannot be activated ${white>
fi
else
        echo -e "${green}server is running ${white}"
fi
}
else
        echo "come with sudo"
 fi
