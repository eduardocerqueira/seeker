#date: 2023-05-03T16:52:45Z
#url: https://api.github.com/gists/0ce083a9e4357ee0a44d997b23da1e2a
#owner: https://api.github.com/users/Frxhb

#!/bin/bash
#
# Script Name: install-apache2-mariadb-php.sh
# Author: github.com/frxhb
# Description: Installs Apache2, MariaDB, and PHP on Ubuntu
# Date: 03.05.2023

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

skip_counter=0

# Check if user is root
if [ $(id -u) -ne 0 ]; then
    echo -e "${RED}This script must be run as root. Try again with 'sudo ./install-apache2-mariadb-php.sh'${NC}\n"
    exit 1
fi

# MariaDB Variables
DB_USER="databaseuser"
DB_PASSWORD= "**********"
DB_NAME="databasename"

function install_apache2 {
  if [ ! -x "$(command -v apache2)" ]; then
    echo -e "${YELLOW}Installing Apache2...${NC}"
    sudo apt-get -y install apache2
    echo -e "${GREEN}Apache2 installation complete.${NC}"
  else
    echo -e "${YELLOW}Apache2 is already installed. Skipping...${NC}"
    skip_counter=$((skip_counter+1))
  fi
}

function install_mariadb {
  if [ ! -x "$(command -v mysql)" ]; then
    echo -e "${YELLOW}Installing MariaDB...${NC}"
    sudo apt-get -y install mariadb-server
    echo -e "${GREEN}MariaDB installation complete.${NC}"

    echo -e "${YELLOW}Creating database and user...${NC}"
    sudo mysql -e "CREATE DATABASE ${DB_NAME};"
    sudo mysql -e "CREATE USER '${DB_USER}'@'localhost' IDENTIFIED BY '${DB_PASSWORD}';"
    sudo mysql -e "GRANT ALL PRIVILEGES ON ${DB_NAME}.* TO '${DB_USER}'@'localhost';"
    echo -e "${GREEN}Database and user creation complete.${NC}"
  else
    echo -e "${YELLOW}MariaDB is already installed. Skipping...${NC}"
    skip_counter=$((skip_counter+1))
  fi
}

function install_php {
  if [ ! -x "$(command -v php)" ]; then
    echo -e "${YELLOW}Installing PHP...${NC}"
    sudo apt-get -y install php libapache2-mod-php php-mysql php-cli
    echo -e "${GREEN}PHP installation complete.${NC}"
  else
    echo -e "${YELLOW}PHP is already installed. Skipping...${NC}"
    skip_counter=$((skip_counter+1))
  fi
}

# Install software based on user input
echo -e "${YELLOW}\nWhich software would you like to install?\n${NC}"
wrong_answers=0
select software in "Apache2" "MariaDB" "PHP" "All"; do
  case $software in
    Apache2)
      install_apache2
      echo -e "\n"
      break
      ;;
    MariaDB)
      install_mariadb
      echo -e "\n"
      break
      ;;
    PHP)
      install_php
      echo -e "\n"
      break
      ;;
    All)
      install_apache2
      install_mariadb
      install_php
      break
      ;;
    *)
      ((wrong_answers++))
      if [[ $wrong_answers -ge 5 ]]; then
        echo -e "${RED}Too many wrong answers. Exiting.${NC}\n"
        break
      fi
      echo -e "${RED}Invalid selection. Please try again.${NC}\n"
      ;;
  esac
done

# Print summary of skipped installations
if [ $skip_counter -gt 0 ]; then
  echo -e "${YELLOW}$skip_counter software installation(s) skipped because it is already installed.${NC}"
fi

# Show database credentials if applicable
if [[ $software == "MariaDB" || $software == "All" ]]; then
  if [ $skip_counter -lt 1 ]; then
    echo -e "${GREEN}\nDatabase credentials:${NC}"
    echo -e "${GREEN}  User:     ${DB_USER}${NC}"
    echo -e "${GREEN}  Password: "**********"
    echo -e "${GREEN}  Database: ${DB_NAME}${NC}"
    echo -e "${YELLOW}\nAttention: These credentials are just examples. If you like to set your own, edit them in the script or re-install mariadb manually and set your own credentials.${NC}"
  else
    echo -e "${RED}Database credentials not shown because MariaDB was already installed.${NC}"
  fi
fi
  fi
fi