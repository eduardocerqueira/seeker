#date: 2024-08-14T18:28:09Z
#url: https://api.github.com/gists/fb89ef02d1b318f672eb731d7fd143f5
#owner: https://api.github.com/users/nmarsceau

# Bash Version

# Shortcut for opening a new Bash shell in a container, if Bash is available. Otherwise, use /bin/sh.
docker-shell () {
  	container="$1"
	  shell='/bin/sh'
	  [ -n $(docker container exec $container which bash) ] && shell='/bin/bash'
	  docker container exec -it $container $shell
}
export -f docker-shell

# Shortcuts for more readable `docker container ls` and `docker image ls` output.
alias dc-ls="docker container ls --format \"table {{.Names}}\t{{.ID}}\t{{.Image}}\t{{.Status}}\""
alias dc-ls-a="docker container ls --all --format \"table {{.Names}}\t{{.ID}}\t{{.Image}}\t{{.Status}}\""
alias di-ls="docker image ls --format \"table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}\t{{.CreatedSince}}\""
