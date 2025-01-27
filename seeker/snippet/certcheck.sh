#date: 2025-01-27T17:11:33Z
#url: https://api.github.com/gists/9033cdd2e9ee57b872a46c9a54867f12
#owner: https://api.github.com/users/chanupen

#!/bin/bash

echo ''$(date)'\n'
echo -e "${YELLOW}1. CertCheck : DomainURL"
echo -e "${YELLOW}2. CertCheck : CertFiles"

#echo -e '\033[1m Choose Options : For Certficate Checks ? : \033[0m' ; read userOption
echo -e "${GREEN} Choose Options : For Certficate Checks ? :" ; read userOption


if [[ $userOption -eq 1 ]]
then 
echo '````````````````````````````````'
echo -e '\033[1m Enter The Domain URL To Check Certficate : \033[0m' ; read domainURL
openssl s_client -servername $domainURL -connect $domainURL:443 | openssl x509 -noout -dates

echo '````````````````````````````````'


elif [[ $userOption -eq 2 ]]
then 
echo '````````````````````````````````'
echo -e '\033[1m Enter The Certificate Filename To Check Certficate : \033[0m' ; read certFilename
cd $(pwd)
openssl x509 -enddate -noout -in $certFilename 2>&1 | grep noAfter | cut -d '=' -f2 

echo '````````````````````````````````'

else
echo -e 'Warning : Please Enter The Correct Option ! '
fi

