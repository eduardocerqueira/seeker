#date: 2021-09-20T16:44:23Z
#url: https://api.github.com/gists/a9aad8d520f92d185b98d52044b1f4ce
#owner: https://api.github.com/users/numeroSette

#!/bin/bash

# https://www.baeldung.com/linux/execute-command-directories

function recursive_for_loop { 
    for f in *;  do 
        if [ -d $f  -a ! -h $f ];  
        then  
            cd -- "$f";  
            echo "Doing something in folder `pwd`/$f"; 

            # use recursion to navigate the entire tree
            recursive_for_loop;
            cd ..; 
        fi;  
    done;  
};
recursive_for_loop