#date: 2024-07-09T17:10:55Z
#url: https://api.github.com/gists/81cdae4248a6a66d62716f40e03ba7f9
#owner: https://api.github.com/users/mabsboza

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 [personal|work1|work2]"
    exit 1
fi

if [ "$1" == "personal" ]; then
    git config --global user.name "Tu Nombre"
    git config --global user.email "tu.email@dominio.com"
elif [ "$1" == "chile" ]; then
    git config --global user.name "Tu Nombre Trabajo 1"
    git config --global user.email "tu.email.trabajo1@dominio.com"
elif [ "$1" == "usa" ]; then
    git config --global user.name "Tu Nombre Trabajo 2"
    git config --global user.email "tu.email.trabajo2@dominio.com"
else
    echo "Invalid option. Use one of [personal|work1|work2]"
    exit 1
fi

echo "Git user configuration updated to $1"