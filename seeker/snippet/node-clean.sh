#date: 2022-01-10T17:21:30Z
#url: https://api.github.com/gists/aaa3811d20c2ed1527e66f10a4572f4d
#owner: https://api.github.com/users/ALazenka

function node-clean() {
	if test -f ./yarn.lock; then
		rm ./yarn.lock
		echo "Removed Yarn Lock"
	fi

	if test -f ./package-lock.json; then
		rm ./package-lock.json
		echo "Removed Package Lock"
	fi

	if test -d ./node_modules; then
		rm -rf ./node_modules
		echo "Removed Node Modules"
	fi
}