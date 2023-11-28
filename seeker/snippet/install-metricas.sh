#date: 2023-11-28T17:02:37Z
#url: https://api.github.com/gists/35ad045c37d9e730a39968b42f16feb4
#owner: https://api.github.com/users/matheusot

# Instala dependencias
sudo apt-get install net-tools
sudo apt-get install vnstat

# Cria arquivo da Crontab
echo "* * * * * sh $HOME/metricas.sh" > "$HOME/crontab.pfg"

# Baixa arquivo metricas
wget https://gist.githubusercontent.com/matheusot/4da9a75a69443a75c54851e57c4cb80d/raw/abec8869edef99859e1eccd328fd1ee45af3bc39/metricas.sh
chmod +x metricas.sh

# Adiciona entrada na crontab do usuario atual
crontab crontab.pfg