#date: 2022-03-01T17:05:15Z
#url: https://api.github.com/gists/c479982b6c2bb9deca6dce144e925269
#owner: https://api.github.com/users/vbalagovic

rootqed="/Users/vedranbalagovic/Projects/qed-local-infrastructure"

project="$rootqed/sources/project/src"

alias qcreate="docker-compose up -d --build --remove-orphans --force-recreate"
alias sz="source ~/.zshrc"

function qedaliases () {
 echo "Qed aliases loaded!"
}

function composer () {
case $PWD in

 "$project/project-api")
   echo -n -e $clear | docker exec --interactive -w /var/www/html/project-api php_7.4_project composer "$@"
   ;;

 *)
   composer "$@"
   ;;
esac
}


function php () {

case $PWD in
 "$project/project-api")
   if [ $1 = "artisan" ]; then
     shift
     docker exec --interactive --workdir /var/www/html/project-api php_7.4_api php artisan "$@"
   else
     docker-compose run --rm -w /var/www/html php "$@"
   fi
   ;;

 *)
   php "$@"
   ;;
esac
}

function npm () {
case $PWD in
 "$monitor/monitor")
   if [ -n "$2" ] && [ $2 = "serve" ]; then
     docker-compose run --rm -p 8185:8185 -w /var/www/html/monitor node run serve -- --port 8185
   else
     docker-compose run --rm -w /var/www/html/monitor node "$@"
   fi
   ;;

 *)
   npm "$@"
   ;;
esac
}