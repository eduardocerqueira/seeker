#date: 2026-01-01T16:56:58Z
#url: https://api.github.com/gists/ebdad0a8e180318e937c3a3e0d497d06
#owner: https://api.github.com/users/kompotkot

#!/usr/bin/env bash
# Home server initialization script for Debian.
#
# Run under root:
# curl -s <script_gist_path_in_raw_mode> | bash

# Colors
C_RESET='\033[0m'
C_RED='\033[1;31m'
C_GREEN='\033[1;32m'
C_YELLOW='\033[1;33m'

# Logs
PREFIX_INFO="${C_GREEN}[INFO]${C_RESET} [$(date +%d-%m\ %T)]"
PREFIX_WARN="${C_YELLOW}[WARN]${C_RESET} [$(date +%d-%m\ %T)]"
PREFIX_CRIT="${C_RED}[CRIT]${C_RESET} [$(date +%d-%m\ %T)]"

# Public keys
PUB_KEY_HS="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCezQaNyflI75XJgnYiHvogbge1A28WTh5p8qimBBiWfcr2cexjAsUi5RnzgdNpnbmf4XNDfNcxA132WAqPKq+R9tng+yRSylSBSvhL8IXnqLitPIGLHvHco9fUJyWlhpVhSDm4fTgsSmoiU2n6Pyt+nh5DxCpuum/tTtn9IBbkXmAXf1tvNBoAVYWCwW4TMysJasH8oqW+RHXWgwJIOnLmjKlEDwpGhgKA9I3RZDSDV5B9h6HlWcAeGrlDL2ibuNDC8e4vKN/6/1ZvDYyWPMkHuotemYyhq+vvMDJ9sfWtZvo1yc20aFZY7zq5SRxYp5vltrzg0kQX/TebJgnk5uGO7mE8Iw4sgcZd0Zbs7mRGCWZNqd4fNr3CfkbgnsHjynbOOFWjEpFZC6S2bp2x2pWCy1Lz0jFCgQW66wyN8PlncGpo8ppzKdBU8wmlU2andvk6/nfMYI0FyL9E7zSflqOMjWnz9nU0g2QcQVH5glThDm22ApoXS7BKmhrFV0gBo/Sk2GsZzyL6OzhW7EWrw0sIM8qZXLe2x7p6B0ZRQ64mIx5b13sMez87wwks58hJAAF0MnCLYl3OqVx+LyAC0zyuIiPj3oHtsrDQ2EwuYxQM2JbgOIogLdLJofleH11JRrEn+meU2ziBR+6H3LDXcAxTiSbGcc63m4QeKFR5X36TUQ== hs_id_rsa"
PUB_KEY_HS_INT="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCqrba1xDD/jBAiEqTVQbo9sbUNDIXhNwQXH1xlvlp6+b32R05JkFF3CTj8qjn8hYqDDl0aJTiYKutKMO3u1jHgmi2QbWVH3eVTy6hGcSqDb4Yn9QxV+RQdiUD/UZplamm+sUGY7zplxQXipWkURrKdPVtdtURLkF5sT2IedlnkQtVOr6LO/JfYJXxpt+Nr3yRsEEvTapMOXYcaCSZhWpRRbbJH+iJwQPGiRnpRneL5vV6eG45tZA2iSnsZDAbTVjf0/wnUzPXdDfcI/kuVToajSdwgVrBoownYb+lCz0xh/caWWwVmqLPtwgiS8Mfmcn2+aa9JD52lnlZi5pWLQr6fqhHqtUZsqGbTiOFSQQEXr+5kupdR9c4jHw+h77TQ0PGjancL+qTqOvFg5ZtQRWeYZGB266dwoau2kahcAoNsHo4+WhuDXrcc0X8ovZ0d1nRhUqD+yJVgVcLPOVpEV2kPemiXAgJousqV1V+j+goGFxd7JH3SQ5zMGuypzh6Ab2+CTh2mGoa5z+FXSScSFLxy2rETtuXA2FaEmY6JZxNiQCWi1bu4SuZdbAJdERIVAVdFDPrHjAy0p5wY3Vr75n81Oy5AN6zRvTuR/+Zh8mCXVtCgqZUGHtprz4BnPogyZs8ZurawuKsGnHTqUp3GnuybNhptBdYyjcFoeb/zqePUTQ== hs_int_id_rsa"
PUB_KEY_HS_VSC="ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCrt40MCupkUhMbbgC3Ou+E0nkGSqpTdfbCz2UVdwIaNWd2/y9KaUNH9jx+dtioBRuwGbpmn3zarNvzsF77/UGOYZZCebCqWbPCcZVZTwo8aTg0BZ+cMGK3OX0D9C1QyVb55c1CDFdtZgPFHD1DPzo/fttEMrUtAdKhPzSK26L1TEAyId7GtZ3e3xdDq6lEs+4VuQhPH5aAkJXOwsoOc1eVw75FpPczOkN7fJ1rI62pvxKEtmNj0JXLvF+wcl09muxeIN/I4qZgus7XEQBfYghnwhaS6j4vEq4kCLHbHJJMIif4Dekb/8uY+dyroWXshxF8BtRrbx46GjmY2+W2zA76hv34ZfyJaP2zRGrb9JQ79fbJaWsgiKmppKyIlRsMiUaROlM24kkGQz/p92JHDNU5SFEVR9IhTAkDLHeB7TlWqzkLMIhmImiztwMBZjWuVxfj1j1uO79lKX6/81I7j+FsQRGOsxY4kHTAcFxRegfaLSsCR9q7q6Pi3ybM/L/WMBFTGeYABpMLCK4hX+qZdU/ZF6ikH4yktPPPIEN7K+sz4y+j1ZLbcd3RlXp2yPMXg5mcQbVneDGIt2IheHJgQ0oB950W8eNPwdCPNtqDQuS157CWWlD+cJuAOXSfFEPg8T4u8cyU04t2KKyTv/4aLE8mnVKYdUP+AlgRsWbkjjV88w== hs_vsc_id_rsa"

USERNAME="${USERNAME:-debian}"
SH_TYPE="${SH_TYPE:-bashrc}"

set -e

if dpkg -l | grep -q "sudo"; then
  echo -e "${PREFIX_INFO} The sudo is installed"
else
  echo -e "${PREFIX_WARN} There is no sudo, installing.."
  apt update
  apt install sudo
fi

# Update visudo privilages
# Grant user NOPASSWD access to sudo
SUDOERS_D_USERS_FILEPATH="/etc/sudoers.d/90-init-users"
touch "${SUDOERS_D_USERS_FILEPATH}"
if grep -P "^${USERNAME}\s+ALL=\(ALL\)\s+NOPASSWD:ALL$" "${SUDOERS_D_USERS_FILEPATH}"; then
  echo -e "${PREFIX_WARN} Sudo NOPASSWD for ${C_YELLOW}${USERNAME}${C_RESET} user already defined"
else
  echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> "${SUDOERS_D_USERS_FILEPATH}"
  echo -e "${PREFIX_INFO} Granted NOPASSWD sudo privileges for ${C_GREEN}${USERNAME}${C_RESET} user"
fi

# Copy SSH keys
SSH_KEYS_FILEPATH="/home/${USERNAME}/.ssh"
if [ -d "$SSH_KEYS_FILEPATH" ]; then
  echo -e "${PREFIX_WARN} Folder for SSH keys already exists"
else
  echo -e "${PREFIX_INFO} Creating folder for SSH keys with path ${SSH_KEYS_FILEPATH}"
  sudo -u "${USERNAME}" mkdir "${SSH_KEYS_FILEPATH}"
  sudo -u "${USERNAME}" chmod 700 "${SSH_KEYS_FILEPATH}"
fi

echo -e "${PREFIX_INFO} Writing SSH public keys to authorized keys file"
echo "${PUB_KEY_HS}" > "${SSH_KEYS_FILEPATH}/authorized_keys"
echo "${PUB_KEY_HS_INT}" >> "${SSH_KEYS_FILEPATH}/authorized_keys"
echo "${PUB_KEY_HS_VSC}" >> "${SSH_KEYS_FILEPATH}/authorized_keys"
chown ${USERNAME}:${USERNAME} "${SSH_KEYS_FILEPATH}/authorized_keys"

# Update SSH config
SSHD_CONFIG_FILEPATH="/etc/ssh/sshd_config"
if [ -f "$SSHD_CONFIG_FILEPATH" ]; then
  echo -e "${PREFIX_INFO} Updating SSH config file ${SSHD_CONFIG_FILEPATH}"
  sed -i -E 's/^\s*#?\s*PasswordAuthentication\s+(yes|no)\s*$/PasswordAuthentication no/' ${SSHD_CONFIG_FILEPATH}
  sed -i -E 's/^\s*#?\s*PermitEmptyPasswords\s+(yes|no)\s*$/PermitEmptyPasswords no/' ${SSHD_CONFIG_FILEPATH}
  sed -i "/^.\{0,2\}PubkeyAuthentication.*/c\PubkeyAuthentication yes" "${SSHD_CONFIG_FILEPATH}"
else
  echo -e "${PREFIX_WARN}Not found config file ${SSHD_CONFIG_FILEPATH}"
fi

echo -e "${PREFIX_INFO} Restarting SSH service"
systemctl restart ssh

# Set tmux config
if [ ! -f "/home/${USERNAME}/.tmux.conf" ]; then
  echo 'set -g default-terminal "screen-256color"' > "/home/${USERNAME}/.tmux.conf"
  echo -e "${PREFIX_INFO} Config for tmux set"
else
  echo -e "${PREFIX_WARN} Tmux config already exists"
fi

# Set ssh-agent startup
if [ ! -f "/home/${USERNAME}/.${SH_TYPE}" ]; then
  echo -e "${PREFIX_WARN} .${SH_TYPE} doesn't exist"
else
  echo '[ -z "$SSH_AUTH_SOCK" ] && eval "$(ssh-agent -s)"' >> "/home/${USERNAME}/.${SH_TYPE}"
  echo -e "${PREFIX_INFO} Config for ssh-agent set"
fi

# Set vim config
sudo -u "${USERNAME}" cat > "/home/${USERNAME}/.vimrc" <<EOF
" Maintainer:   kompotkot
" Last change:  2021 May 10
"
" Get the defaults that most users want.
" source $VIMRUNTIME/defaults.vim

filetype plugin indent on       " recognize file extensions

set history=200
set ruler           " display current postion of cursor
set showcmd         " show incomplete command
set number          " show line numbers
set expandtab       " replace tabs with spaces
set tabstop=4       " ir tab 4 spaces
set shiftwidth=4    " 4 spaces width when identing with '>'

set mouse=          " disable mouse
set ttymouse=

syntax on

" netrw - tree view
let g:netrw_liststyle = 1       " view with dates
let g:netrw_banner = 0          " hide banner
let g:netrw_browse_split = 4    " open in previous window
let g:netrw_winsize = 15        " width of tree view

" When started as "evim", evim.vim will already have done these settings, bail
" out.
if v:progname =~? "evim"
  finish
endif

if &t_Co > 2 || has("gui_running")
  " Switch on highlighting the last used search pattern.
  set hlsearch
  set incsearch     " highlight current search
endif

" Add optional packages.
"
" The matchit plugin makes the % command work better, but it is not backwards
" compatible.
" The ! means the package won't be loaded right away but when plugins are
" loaded during initialization.
if has('syntax') && has('eval')
  packadd! matchit
endif
EOF
echo -e "${PREFIX_INFO} Config for vim set"