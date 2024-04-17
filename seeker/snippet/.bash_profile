#date: 2024-04-17T17:06:57Z
#url: https://api.github.com/gists/97293c922bb81aa2ffd74a9cc654c03b
#owner: https://api.github.com/users/tiagobalsas

# No macOS, o arquivo ~/.bash_profile é executado para shells de login interativas, enquanto o ~/.bashrc é executado para shells interativas não-login. Quando você abre um novo terminal no macOS, ele inicia um shell de login, então as alterações feitas em ~/.bash_profile serão aplicadas.
# Portanto, você deve adicionar os comandos ao arquivo ~/.bash_profile. Aqui estão os passos:
# Abra o terminal.
# Digite open -e ~/.bash_profile para abrir o arquivo em um editor de texto.
# Adicione o seguinte ao final do arquivo:

# Inicia o ssh-agent
if [ -z "$(pgrep ssh-agent)" ]; then
    eval "$(ssh-agent -s)"
fi

# Adiciona a chave ao ssh-agent
if [ ! -z "$(pgrep ssh-agent)" ]; then
    ssh-add ~/.ssh/id_rsa
fi

# Salve e feche o arquivo.
# Para que as alterações entrem em vigor, você pode digitar source ~/.bash_profile no terminal.