#date: 2025-12-04T16:57:24Z
#url: https://api.github.com/gists/0f9b9d85c7d9a093e3628734a5276998
#owner: https://api.github.com/users/greenaj

#!/usr/bin/env bash

mkdir -p ~/.vim/{backup,swp,undo}

cat << EOF > ~/.vimrc
set nocompatible
filetype off
set modelines=5
set backupdir=~/.vim/backup//
set directory=~/.vim/swp//
set undodir=~/.vim/undo//
filetype plugin indent on
syntax on
set et sts=4 sw=4 ts=4
autocmd FileType html setlocal expandtab shiftwidth=2 tabstop=2
autocmd FileType markdown setlocal expandtab shiftwidth=2 tabstop=2
EOF