#date: 2023-04-20T17:03:59Z
#url: https://api.github.com/gists/2ddb5559b0ca6636d8cca3d095e374a7
#owner: https://api.github.com/users/shivaram235vemula

#replace EBS Asserter public ip with actual value. 
docker run -d --name asrPOCserver1 --link wlsadmin: "**********":7003 -p 30018:7018 -v /u01/oracle/properties/domain.properties:/u01/oracle/properties -e MS_NAME=asrPOCserver1 wls:12.2.1.3.0 /u01/oracle/Middleware/domains/asr_domain/bin/startManagedWebLogic.sh asrPOCserver1 http://<EBS Asserter Public IP Address>:7001 -Dweblogic.management.allowPasswordEcho=true.allowPasswordEcho=true