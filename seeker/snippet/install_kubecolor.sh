#date: 2022-04-28T17:01:48Z
#url: https://api.github.com/gists/4a020711968e875b3df40a24f4a0e7e1
#owner: https://api.github.com/users/VictorGil-Ops

# https://github.com/hidetatz/kubecolor

# from: https://github.com/hidetatz/kubecolor/releases
_kubeclor_ver="0.0.20"

## FEDORA
sudo yum install bash-completion -y;
source ~/.bashrc;
type _init_completion;
source /usr/share/bash-completion/bash_completion;
echo 'source <(kubectl completion bash)' >>~/.bashrc;
wget https://github.com/hidetatz/kubecolor/releases/download/v${_kubeclor_ver}/kubecolor_${_kubeclor_ver}_Linux_x86_64.tar.gz;
tar -xvf kubecolor_${_kubeclor_ver}_Linux_x86_64.tar.gz;
sudo cp -a kubecolor /bin/;
complete -o default -F __start_kubectl kubecolor;
complete -o default -F __start_kubectl kc
echo 'alias kubectl="kubecolor"' >> ~/.bashrc;
echo 'alias kc="kubectl"' >> ~/.bashrc;
source ~/.bashrc;