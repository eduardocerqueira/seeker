#date: 2021-12-23T17:20:50Z
#url: https://api.github.com/gists/5288a29a9336c7d123abed745f482006
#owner: https://api.github.com/users/onuncukoy-dot

#!/bin/bash

# sudo ln -s $PWD/naabu-scan.sh /usr/local/bin/naabu-scan

GREEN="\033[0;32m"
NC='\033[0m'

echo -e "${GREEN}Creating folder naabu-output/${NC}"
mkdir naabu-output

echo -e "${GREEN}-=- Scanning -=-${NC}"
echo -e "${GREEN}Ports: 445${NC}"
naabu -silent -iL scope.txt -p 445 -o naabu-output/naabu-smb.out
echo -e "${GREEN}Ports: 21${NC}"
naabu -silent -iL scope.txt -p 21 -o naabu-output/naabu-ftp.out
echo -e "${GREEN}Ports: 389,636,3268,3269${NC}"
naabu -silent -iL scope.txt -p 389,636,3268,3269 -o naabu-output/naabu-ldap.out
echo -e "${GREEN}Ports: 9200,6379,5601${NC}"
naabu -silent -iL scope.txt -p 9200,6379,5601 -o naabu-output/naabu-elastic-redis-kibana.out
echo -e "${GREEN}Ports: 2375,2376,5000,51678${NC}"
naabu -silent -iL scope.txt -p 2375,2376,5000,51678 -o naabu-output/naabu-docker.out
echo -e "${GREEN}Ports: 1098,1099,1050${NC}"
naabu -silent -iL scope.txt -p 1098,1099,1050 -o naabu-output/naabu-rmi.out
echo -e "${GREEN}Ports: 2049${NC}"
naabu -silent -iL scope.txt -p 2049 -o naabu-output/naabu-nfs.out
echo -e "${GREEN}Ports: 3306,1433,5432,5433,5984,6984,9001,27017,27018${NC}"
naabu -silent -iL scope.txt -p 3306,1433,5432,5433,5984,6984,9001,27017,27018 -o naabu-output/naabu-sql.out
echo -e "${GREEN}Ports: 9000,10000${NC}"
naabu -silent -iL scope.txt -p 9000,10000 -o naabu-output/9-10000.out
echo -e "${GREEN}Ports: 25,465,587${NC}"
naabu -silent -iL scope.txt -p 25,465,587 -o naabu-output/naabu-smtp.out
echo -e "${GREEN}Ports: 143,993${NC}"
naabu -silent -iL scope.txt -p 143,993 -o naabu-output/naabu-imap.out
echo -e "${GREEN}Ports: 22${NC}"
naabu -silent -iL scope.txt -p 22 -o naabu-output/naabu-ssh.out