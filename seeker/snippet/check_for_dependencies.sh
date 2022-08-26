#date: 2022-08-26T16:46:08Z
#url: https://api.github.com/gists/e3da90c93d787c844da81642182acb07
#owner: https://api.github.com/users/arbal

# Easy way to check for dependencies
checkfor () {
    command -v $1 >/dev/null 2>&1 || { 
        echo >&2 "$1 required"; 
        exit 1; 
    }
}
checkfor "ffmpeg"


# example using an array of dependencies
array=( "convert" "ffmpeg" )
for i in "${array[@]}"
do
    command -v $i >/dev/null 2>&1 || { 
        echo >&2 "$i required"; 
        exit 1; 
    }
done