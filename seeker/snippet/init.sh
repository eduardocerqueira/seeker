#date: 2022-03-01T17:01:34Z
#url: https://api.github.com/gists/61295525d30829f1286b6fd7c7655629
#owner: https://api.github.com/users/vbalagovic

#!/bin/bash
rootpwd=$PWD
 
startNetworks() {
   printf "$(tput setaf 2)✔ Starting docker networks...\n"
    docker network create databases_qed_infrastructure > /dev/null 2>&1
    docker network create project_1_app_1 > /dev/null 2>&1
    docker network create monitor > /dev/null 2>&1
    docker network create applications > /dev/null 2>&1
   printf "$(tput setaf 2)\n✔ Docker networks started!\n"
}
 
startMonitorDockers() {
    printf "$(tput setaf 2)✔ Starting MONITOR project dockers\n"
    cd "$rootpwd/sources/monitor/src"
     docker-compose up -d --build --remove-orphans --force-recreate
    printf "$(tput setaf 2)\n✔ MONITOR containers started!\n"
}

runMonitor() {
    printf "$(tput setaf 2)✔ Running MONITOR commands\n"
    cd "$rootpwd/sources/monitor/src/monitor"
     docker-compose run --rm -w //var/www/html/monitor node install
     docker-compose run --rm -w //var/www/html/monitor node rebuild node-saas
     docker-compose run --rm -w //var/www/html/monitor node run build
    printf "$(tput setaf 2)✔ All set!\n"
}

createProject1App1() {
   printf "$(tput setaf 2)✔ Cloning Project 1 App 1"
   cd "$rootpwd/sources/project_1/src"
   git clone ssh://directory
   cd "$rootpwd/sources/project_1/src/project_1_app_1"
   cp "$rootpwd/sources/project_1/.env.project_1_app_1" .env
   printf "$(tput setaf 2)✔ Cloned!"
}
 
runProject1App1() {
   printf "$(tput setaf 2)✔ Installing packages Project 1 App 1"
   cd "$rootpwd/sources/project_1/src"
   docker-compose up -d --build --remove-orphans --force-recreate
   docker-compose run --rm -w //var/www/html/project_1_app_1 node install
   printf "$(tput setaf 2)✔ Packages installed!"
   printf "$(tput setaf 2)✔ Building Project 1 App 1"
   docker-compose run --rm -w //var/www/html/project_1_app_1 node run build
   docker-compose run --rm -w //var/www/html/project_1_app_1 node run generate
   printf "$(tput setaf 2)✔ Project 1 App 1 built!"
}
   
run() {
if [[ $# -eq 0 ]] ; then
       echo "No arguments supplied"
   fi
 
   for var in "$@"
   do
       case $var in
           "networks")
               startNetworks & spinner
               ;;
               
           "monitor")
                startMonitorDockers & spinner
                runMonitor & spinner
                ;;
 
           "project_1_app_1")
               createProject1App1
               runProject1App1
               ;;
           *)
               printf "Parameter not recognized: $var"
               ;;
       esac
   done
}
 
run "$@"

