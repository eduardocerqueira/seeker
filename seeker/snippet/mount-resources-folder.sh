#date: 2022-08-26T16:59:04Z
#url: https://api.github.com/gists/5ce57b910775e18a25e31884d2c57fa7
#owner: https://api.github.com/users/marcoscssno

#!/bin/bash

# Script para montar pasta de recursos para configuração de sistema operacional
# Desenvolvido por Marcos Cassiano Melo Feijão
# Em 26 de agosto de 2022
# Em Sobral - Ceará - Brasil

APPLICATION_NAME=mount-resources-folder
VERSION=1.0

echo $APPLICATION_NAME $VERSION

check_administrator_user()
{
    if [ -z $ADMINISTRATOR_USER ]
    then
        echo "Você ainda não definiu um usuário Administrador"
    else
        echo "O usuário Administrador é ${ADMINISTRATOR_USER}"
    fi
}

check_administrator_user

while :
do
    echo "O usuário Administrador é `whoami`? (^C para cancelar)"
    read X
    case $X in
        s)
            ADMINISTRATOR_USER=`whoami`
            check_administrator_user
            break;;
        n)
            echo "Qual é o nome do usuário Administrador?"
            read X
            ADMINISTRATOR_USER=$X
            check_administrator_user
            break;;
        *)
            echo "Opção inválida";
    esac
done

echo "Qual é o sistema de arquivos?"
echo "Exemplo: //10.0.0.1/shared_folder"
read X
FILE_SYSTEM=$X

echo sudo mkdir -v /home/$ADMINISTRATOR_USER/mint
echo sudo mkdir -v /root/credentials
echo sudo nano /root/credentials/.smbpasswd1
echo "# mint (${ADMINISTRATOR_USER})" > config.txt
echo "${FILE_SYSTEM} /home/${ADMINISTRATOR_USER}/mint cifs credentials=/root/credentials/.smbpasswd1,uid=${ADMINISTRATOR_USER},iocharset=utf8,perm 0 0" >> config.txt
xed -w config.txt
echo sudo nano /etc/fstab
echo "Início de teste de montagem"
echo sudo mount -f /home/$ADMINISTRATOR_USER/mint
echo "Fim de teste de montagem"
while :
do
    echo "Prosseguir com a montagem acima? (s/n) (^C para cancelar)"
    read X
    case $X in
        s)
            break;;
        n)
            echo "Por favor, tente novamente"
            exit;;
        *)
            echo "Opção inválida";
    esac
done
echo sudo mount -v /home/$ADMINISTRATOR_USER/mint
echo thunar /home/$ADMINISTRATOR_USER/mint
rm -v config.txt
exit