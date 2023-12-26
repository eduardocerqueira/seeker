#date: 2023-12-26T17:06:33Z
#url: https://api.github.com/gists/383430a6298f9665782e864877959294
#owner: https://api.github.com/users/pabsan-0

#!/bin/bash

# Terminology:
#
#  ~                                       # target_dir
#  ├── .tmux.conf -> dotfiles/vim/.vimrc   # target_link  target_file
#  ├── .vimrc
#  │ 
#  └── dotfiles                            # stow_dir
#      ├── tmux                            # stow_pkg
#      │   └── .tmux.conf                  # stow_file
#      └── vim
#          └── .vimrc


stow_dir="$HOME/dotfiles"
target_dir="$HOME"

# Directories to ignore within stow_dir (separated by | for extended regex)
ignore_dirs="code|.git|.tools|.docs"

# Function to check if a file is a symlink
is_symlink() {
    [ -L "$1" ]
}

# Function to compare dotfiles with system files
compare_files() {
    
    # Find and loop all packages in the stow_dir
    stow_pkgs=$(find "$stow_dir" -mindepth 1 -maxdepth 1 -type d | grep -E -v "($ignore_dirs)")
    for stow_pkg in ${stow_pkgs[@]}; do

        stow_pkg_rel=${stow_pkg#$stow_dir/}

        # Find and loop all files in the stow_pkg
        # stow_files=($(find "$stow_pkg" -type f | grep -E -v "($ignore_dirs)"))

        mapfile -d '' -t stow_files < <(find $stow_pkg -type f -print0)
        for stow_file in "${stow_files[@]}"; do

            # Get name of file relative to stow_dir and figure where target_link ends up being
            stow_file_rel=${stow_file#$stow_dir/*/}
            target_link="$target_dir/$stow_file_rel"
            target_file=$(readlink -f $target_link)

            if [ -e "$target_link" ]; then
                if is_symlink "$target_link"; then
                    if [[ "$target_file" == "$stow_pkg/$stow_file_rel" ]]; then
                        echo -e "\e[33m$stow_pkg_rel \t$stow_file_rel\e[0m \t(Symlink exists in system, points to dotfile)"
                    else
                        echo -e "\e[35m$stow_pkg_rel \t$stow_file_rel\e[0m \t(Symlink exists in system, points elsewhere)"
                    fi
                else
                    echo -e "\e[32m$stow_pkg_rel \t$stow_file_rel\e[0m \t(File exists in system)"
                fi
            else
                echo -e "\e[34m$stow_pkg_rel \t$stow_file_rel\e[0m \t(Not stowed)"
            fi

        done
    done

}

compare_files | column -ts $'\t'
