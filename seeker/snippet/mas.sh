#date: 2024-12-12T17:10:08Z
#url: https://api.github.com/gists/1fa5745ab8d20a4016edc926d812129a
#owner: https://api.github.com/users/Justman100

#!/bin/bash

echo "ACME.sh Manager"
sleep 1
echo "By Justman10000"
sleep 2.5

note() {
    echo "BEWARE! Running webservers on port 80 must be deactivated"
    sleep 1
}

! [[ $HOME ]] && export HOME=/root

if ! [[ -d $HOME/.acme.sh ]]; then
    echo "ACME.sh is not installed!"
    sleep 1.5
    echo "Do you want install it?"
    read -p "" install
elif [[ ! -f acme.sh ]]; then
    echo "You must run this script in the same directory"
fi

case $1 in
    -i|--install)
        if [[ -d $HOME/.acme.sh ]]; then
            echo "ACME.sh is already installed"
            exit 1
        fi

        wget https://raw.githubusercontent.com/acmesh-official/acme.sh/refs/heads/master/acme.sh
        mkdir /etc/letsencrypt

        bash acme.sh --install --cert-home /etc/letsencrypt/live
        ln -fs $HOME/.acme.sh /home/acme
        exit
    ;;

    -s|--standalone)
        note
        bash acme.sh --server letsencrypt --issue --domain $2 --standalone \
                --cert-file /etc/letsencrypt/live/${2}_ecc/cert.pem \
                --key-file /etc/letsencrypt/live/${2}_ecc/privkey.pem \
                --fullchain-file /etc/letsencrypt/live/${2}_ecc/fullchain.pem
    ;;

    -w|--wildcard)
        note
        bash acme.sh --server letsencrypt --issue --domain *.$2 --dns --yes-I-know-dns-manual-mode-enough-go-ahead-please \
                --cert-file /etc/letsencrypt/live/${2}_ecc/cert.pem \
                --key-file /etc/letsencrypt/live/${2}_ecc/privkey.pem \
                --fullchain-file /etc/letsencrypt/live/${2}_ecc/fullchain.pem
    ;;

    -r|--renew)
        note
        bash acme.sh --server letsencrypt --renew --domain *.$2 --dns --yes-I-know-dns-manual-mode-enough-go-ahead-please
    ;;

    *)
        echo "Unknown argument"
        exit 1
    ;;
esac