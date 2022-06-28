#date: 2022-06-28T16:58:39Z
#url: https://api.github.com/gists/8053f0804a694c34cb18cd4035e0993c
#owner: https://api.github.com/users/ippei8jp

cd ~
wget https://gist.githubusercontent.com/ippei8jp/8179edb10867faf98e233a52965a9e53/raw/4f39afbcd8471426421944b597f3a5f2963984c6/resize.py
chmod +x resize.py

cp .bashrc .bashrc.org
cat >> .bashrc << _EOF_
# プロンプトの設定
PS1="\w\$ "

# キーバインドの設定
bind '"\C-n": history-search-forward'
bind '"\C-p": history-search-backward'

# ディレクトリスタックの表示改善
function pushd() {
    command pushd $* > /dev/null
    command dirs -v
}
function popd() {
    command popd $* > /dev/null
    command dirs -v
}
function dirs() {
    command dirs -v
}

# 表示色変更
export LS_COLORS='di=01;32:ln=01;36:ex=01;31:'
export GREP_COLORS='mt=01;31:ml=:cx=:fn=01;32:ln=32:bn=32:se=36'

# lessのオプション
export LESS="-iMR"

# reset console size
case "$TERM" in
    vt220) ~/resize.py ;;
esac

# pyenv 設定
export PYENV_ROOT=/proj/.pyenv
if [ -e $PYENV_ROOT ]; then
    export PATH=$PYENV_ROOT/bin:$PATH
    # Raspbian向け対策(numpyでundefined symbol: PyFPE_jbuf)
    export PYTHON_CONFIGURE_OPTS="\
     --enable-ipv6\
     --enable-unicode=ucs4\
     --enable-shared\
     --with-dbmliborder=bdb:gdbm\
     --with-system-expat\
     --with-system-ffi\
     --with-fpectl"

    eval "$(pyenv init --path)"          # pyenv 2.0以降で必要
    eval "$(pyenv init -)"
    eval "$(pyenv virtualenv-init -)"
fi

# nodenv 設定
export NODENV_ROOT=/proj/.nodenv
if [ -e $NODENV_ROOT ]; then
    export PATH=$NODENV_ROOT/bin:$PATH
    eval "$(nodenv init -)"
fi

_EOF_

# inputrcの編集
sudo cp /etc/inputrc /etc/inputrc.org
sudo cat >> /etc/inputrc << _EOF_
# disable bracked-paste mode
set enable-bracketed-paste off
_EOF_

# keyboardマッピングの変更
sudo cp /etc/default/keyboard /etc/default/keyboard.org
sudo sed -i "s/^XKBOPTIONS.*/XKBOPTIONS=\"ctrl:swapcaps\"      # CapsLock <-> Ctrl/g" /etc/default/keyboard

# ワークディレクトリの作成
sudo mkdir /work
sudo mkdir /proj
sudo chown $USER:$USER /work /proj

# sambaのインストール
sudo apt install -y samba
sudo smbpasswd -a $USER
# パスワード入力

# smb.confの編集
sudo sed -i "/\[homes\]/,/^\[\|^;\s*\[/ s/read only = .*/read only = no/1" /etc/samba/smb.conf 
sudo sed -i 's/\(^\[global\].*\)/\1\n\n    map archive = n/' /etc/samba/smb.conf
sudo cat >> /etc/samba/smb.conf << _EOF_
[work]
path = /work
guest ok = no
writable = yes
map archive = no
share modes = yes
dos filetimes = yes
force group = $USER
force create mode = 0664

[proj]
path = /proj
guest ok = no
writable = yes
map archive = no
share modes = yes
dos filetimes = yes
force group = $USER
force create mode = 0664

_EOF_

# sambaの再起動
sudo service smbd reload
sudo service smbd restart


# VNCの有効化
raspi-config do_vnc 0
sudo raspi-config nonint do_vnc_resolution 1920x1080
