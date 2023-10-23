#date: 2023-10-23T16:57:22Z
#url: https://api.github.com/gists/d3a2346c2bfc7388c07c060b0c015e64
#owner: https://api.github.com/users/faelpinho


# Mudar para zsh
chsh -s $(which zsh)

# Aumentar downloads paralelos do pacman e pamac
sudo nano /etc/pacman.conf
sudo nano /etc/pamac.conf

# Adicionar $USER ao grupo
sudo usermod -aG tty $USER

# Descompactar
unzip -d FILE.zip
unrar x FILE.rar
unxz -d FILE.xz

# Caçar os problemas
journalctl -b 0 | grep bluetooth --color

# Verificar integridade de 2 arquivos
sha256sum file1 file2

# Converter imagem para formato aceito no PS Vita (bg e startup)
pngquant --posterize 4 image.png

# Ver só 2 linhas de arquivo (cat/tac)
cat -n 2 file.log

# Conexão remota
xfreerdp /d:dominio /cert-ignore /u:username /v:hostname /w:1920 /h:1080 /smart-sizing:1920x1080 /monitors:0

# Teste rápido php
php -S localhost:8000 --docroot=src --php-ini=php.ini -a # >& /dev/null

