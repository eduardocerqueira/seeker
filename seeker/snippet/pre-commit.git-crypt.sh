#date: 2023-06-16T17:01:30Z
#url: https://api.github.com/gists/b2896351a82a9877706310e0912030f3
#owner: https://api.github.com/users/jimbrig

#!/bin/bash
# Time-stamp: <Tue 2021-08-31 14:26 svarrette>
################################################################################
# Pre-commit hook to avoid accidentally adding unencrypted files with
# [git-crypt](https://www.agwa.name/projects/git-crypt/)
# Fix to [Issue #45](https://github.com/AGWA/git-crypt/issues/45)
#
# Usage:
#    $ cd /path/to/repository
#    $ git-crypt init
#    $ curl <url/to/this/raw/gist> -o .git/hooks/pre-commit
#    $ chmod +x .git/hooks/pre-commit
#
# Otherwise, you might want to add it as a git submodule, using:
#    $ git submodule add https://gist.github.com/848c82daa63710b6c132bb42029b30ef.git config/hooks/pre-commit.git-crypt
#    $ cd .git/hooks
#    $ ln -s ../../config/hooks/pre-commit.git-crypt/pre-commit.git-crypt.sh pre-commit
#
# The latest version of this script can be found as a GIST on:
#          https://gist.github.com/Falkor/848c82daa63710b6c132bb42029b30ef
#
################################################################################
# MIT License
#
# Copyright (c) 2016-2021 S. Varrette <Sebastien.Varrette@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################
#
if [ -d .git-crypt ]; then
    STAGED_FILES=$(git diff --cached --name-status | awk '$1 != "D" { print $2 }' | xargs echo)
    if [ -n "${STAGED_FILES}" ]; then
        git-crypt status ${STAGED_FILES} &>/dev/null
        if [[ $? -ne 0  ]]; then
            git-crypt status -e ${STAGED_FILES}
            echo '/!\ You should have first unlocked your repository BEFORE staging the above file(s)'
            echo '/!\ Proceed now as follows:'
            echo -e "\t git unstage ${STAGED_FILES}"
            echo -e "\t git crypt unlock"
            echo -e "\t git add ${STAGED_FILES}"
            exit 1
        fi
    fi
fi
