#date: 2022-07-20T17:02:26Z
#url: https://api.github.com/gists/2cce81322d00059d2e3ec902ba9a18b4
#owner: https://api.github.com/users/tlcd96

#!/bin/bash

# ln -s /etc/php/global/php.ini /etc/php/*/*/conf.d/99_global.ini

Acceptable=("apache2" "cli" "fpm")

for i in $(cd /etc/php/ && ls -d *); do
        if [ "$i" != "global" ]; then
                for e in "${Acceptable[@]}"
                do
                        DIRCHECK="/etc/php/${i}/${e}/conf.d"
                        if [[ -d "$DIRCHECK" ]]; then
                                FILESYMLINK="$DIRCHECK/99_global.ini"
                                if [[ ! -f "${FILESYMLINK}" ]]; then
                                        ln -s /etc/php/global/php.ini "${FILESYMLINK}"
                                fi;
                        fi
                done
        fi
done