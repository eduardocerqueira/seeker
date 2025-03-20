#date: 2025-03-20T16:58:52Z
#url: https://api.github.com/gists/e7d8e6afab182683903001e54f454ec6
#owner: https://api.github.com/users/washopilot

docker run --rm -it \
  -v "$(pwd):/workspace" \
  -w /workspace \
  php:8.4-cli-bookworm bash -c "apt update && apt install -y sudo curl unzip && groupadd -g $(id -g) devgroup && useradd -m -u $(id -u) -g $(id -g) dev && echo 'dev ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && sudo -u dev bash -c 'curl -sS https://getcomposer.org/installer | php && sudo mv composer.phar /usr/local/bin/composer && composer global require laravel/installer && echo \"export PATH=\\\$HOME/.composer/vendor/bin:\\\$PATH\" >> ~/.bashrc && exec bash'"
