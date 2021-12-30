#date: 2021-12-30T16:51:31Z
#url: https://api.github.com/gists/1d0b531325ff3613f6daad271a55bbba
#owner: https://api.github.com/users/anikolaienko

# Accept either empty path or valid path

while [ -z $path_confirm ]; do
    read -rp 'Type folder path (relative to current):' folder_path

    path_confirm="y"

    if [ ! -z $folder_path ] && [[ ! -d $folder_path ]]; then
        read -rp 'Path does not exist. Do you want to type it again? [y/n]:' path_confirm

        if [[ $path_confirm = "y" ]] || [[ $path_confirm = "Y" ]]; then
            unset path_confirm
        fi
    fi
done

echo "Result path: [$(pwd)/$folder_path]"
