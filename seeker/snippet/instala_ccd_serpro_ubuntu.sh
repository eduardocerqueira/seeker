#date: 2024-09-03T16:48:17Z
#url: https://api.github.com/gists/2b409571e3939324a53e30c8ffad8b5d
#owner: https://api.github.com/users/rodrigomaia

# Entrar na pasta de certificados do sistema:
# Ubuntu anterior ao 18:
sudo mkdir /usr/share/ca-certificates/serpro/
sudo cd /usr/share/ca-certificates/serpro/
# Ubuntu 18:
sudo mkdir /usr/local/share/ca-certificates/serpro/
cd /usr/local/share/ca-certificates/serpro/

# Baixar os certificados do repositorio:
wget -r --no-check-certificate https://ccd.serpro.gov.br/serproacf/docs/

# Remover apenas os certificados de interesse:
find ./ccd.serpro.gov.br -name *.crt | sudo xargs -I{} cp -u {} .

# Limpar o restante do wget:
sudo rm -rf ccd.serpro.gov.br/

# Executar compilação dos certificados para o sistema:
sudo update-ca-certificates

# Instalar os certificados nos browsers
sudo apt install libnss3-tools

# Instalar os certificados no google-chrome
for i in $(ls /usr/local/share/ca-certificates/serpro/); do $(certutil -d sql:$HOME/.pki/nssdb -A -t "C,C,C" -n $i -i /usr/local/share/ca-certificates/serpro/$i); done

# Instalar os certificados no firefox
cat ~/.mozilla/firefox/profiles.ini 

# Anote o valor do Default no meu caso: yxsfy966.default-release
# [Install4F96D1932A9F858E]
# Default=yxsfy966.default-release
# Locked=1

# O Resultado você subistitui no comando abaixo logo depois do $HOME/.mozilla/firefox/:    
for i in $(ls /usr/local/share/ca-certificates/serpro/); do $(certutil -d sql:$HOME/.mozilla/firefox/yxsfy966.default-release -A -t "C,C,C" -n $i -i /usr/local/share/ca-certificates/serpro/$i); done