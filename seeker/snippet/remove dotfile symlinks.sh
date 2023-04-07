#date: 2023-04-07T16:46:07Z
#url: https://api.github.com/gists/20b0f616b9503204d9147b17f2a03e6c
#owner: https://api.github.com/users/Finkregh

# this expects your dotfiles in .dotfiles

for l in $(
	for link in $(find . -type l); do
		if readlink "$link" | grep -q ".dotfiles"; then
			echo "$link"
		fi
	done
	); do
	echo /bin/cp --remove-destination "$(realpath "$l")" "$l"
done | tee symlinks-todo.sh

# then look at symlinks-todo.sh and if content is ok run it
