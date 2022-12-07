#date: 2022-12-07T17:02:09Z
#url: https://api.github.com/gists/de622beffec23b2dad42395475ffbdf0
#owner: https://api.github.com/users/lfelipe1501

#!/bin/bash

#
# Script to Install oh-my-posh in RHEL Base Linux Distribution.
# @author   Luis Felipe <lfelipe1501@gmail.com>
# @website  https://www.lfsystems.com.co
# @version  1.0

#Color variables
W="\033[0m"
R="\033[01;31m"
OB="\033[44m"

case "$1" in
install)
if ! [ -x "$(command -v oh-my-posh)" ]; then

if ! [ -x "$(command -v pwsh)" ]; then
	echo ""
	echo "You should check the version available for your RHEL linux distribution"
	yld=$(rpm -q --provides $(rpm -q --whatprovides "system-release(releasever)") | grep "system-release(releasever)")
	echo -e "You have the operating system:$R $yld$W"
	echo "This can be done from the official Microsoft page:"
	echo -e "$OB https://packages.microsoft.com/config/rhel/$W"
	echo "Indicate number of Microsoft REPO to be used, Example: 8"
	echo -n "> "
	read rhlver

	## Create Folder for PWSH profile
	mkdir -p ~/.config/powershell

	## Install PowerShell from official repo.
	wget https://packages.microsoft.com/config/rhel/$rhlver/prod.repo -O /tmp/microsoft.repo && sudo cp /tmp/microsoft.repo /etc/yum.repos.d/ && sudo yum install --assumeyes powershell
fi

## Install oh-my-posh to use with pwsh or any shell
sudo wget -q https://github.com/JanDeDobbeleer/oh-my-posh/releases/latest/download/posh-linux-amd64 -O /usr/local/bin/oh-my-posh && sudo chmod +x /usr/local/bin/oh-my-posh

mkdir ~/.poshthemes

wget https://github.com/JanDeDobbeleer/oh-my-posh/releases/latest/download/themes.zip -O ~/.poshthemes/themes.zip && unzip ~/.poshthemes/themes.zip -d ~/.poshthemes && chmod u+rw ~/.poshthemes/*.omp.* && rm ~/.poshthemes/themes.zip

sudo cat > /tmp/ohmyposh.sh <<EOF
export POSH_THEMES_PATH=$HOME/.poshthemes
EOF

sudo cat >> /tmp/ohmyposh.sh <<'EOF'
export PATH=$PATH:$POSH_THEMES_PATH
EOF

sudo cp /tmp/ohmyposh.sh /etc/profile.d/

pwsh -Command Set-PSRepository -Name 'PSGallery' -InstallationPolicy Trusted
pwsh -Command Install-Module -Name PowerShellGet -Force -Scope CurrentUser
pwsh -Command Install-Module MagicTooltips -Scope CurrentUser
pwsh -Command Install-Module PSReadLine -Force -Scope CurrentUser
pwsh -Command Install-Module -Name Terminal-Icons -Repository PSGallery -Scope CurrentUser

cat > ~/.config/powershell/Microsoft.PowerShell_profile.ps1 <<'EOF'
oh-my-posh init pwsh --config "$env:POSH_THEMES_PATH\atomic.omp.json" | Invoke-Expression
Enable-PoshTooltips

# Update All Modules
function Update-EveryModule {
   Set-PSRepository -Name 'PSGallery' -InstallationPolicy Trusted
   #$PSDefaultParameterValues = @{‘Install-Module:Scope’=’AllUsers’; ‘Update-Module:Scope’=’AllUsers’}
   Get-InstalledModule | Update-Module -Verbose
}

# Remove Old Modules
function Remove-OldModules {
        $Latest = Get-InstalledModule
        foreach ($module in $Latest) {
                Write-Verbose -Message "Uninstalling old versions of $($module.Name) [latest is $( $module.Version)]" -Verbose
                Get-InstalledModule -Name $module.Name -AllVersions | Where-Object {$_.Version -ne $module.Version} | Uninstall-Module -Verbose
        }
}

# Enable Icons CMD
Import-Module -Name Terminal-Icons

# Enable Predictive IntelliSense
Import-Module PSReadLine
Set-PSReadLineOption -PredictionSource History
Set-PSReadLineOption -PredictionViewStyle ListView
Set-PSReadLineOption -EditMode Windows

EOF

chsh -s $(which pwsh)

else
	echo "oh-my-posh "$(oh-my-posh version)" is already installed!"
	echo -e "If you want you can run \e[42$0 update\e[0m to check for updates"
	exit
fi
   ;;
update)
   actual=$(oh-my-posh version)

   wget -q https://github.com/JanDeDobbeleer/oh-my-posh/releases/latest/download/posh-linux-amd64 -O /tmp/ohmyposh && chmod +x /tmp/ohmyposh

   nwv=$(/tmp/ohmyposh version)

   if [ "$actual" = "$nwv" ]; then
	echo "The program is already updated! =)"
   else
    sudo wget -q https://github.com/JanDeDobbeleer/oh-my-posh/releases/latest/download/posh-linux-amd64 -O /usr/local/bin/oh-my-posh && sudo chmod +x /usr/local/bin/oh-my-posh

    wget https://github.com/JanDeDobbeleer/oh-my-posh/releases/latest/download/themes.zip -O ~/.poshthemes/themes.zip && unzip -o ~/.poshthemes/themes.zip -d ~/.poshthemes && chmod u+rw ~/.poshthemes/*.omp.* && rm ~/.poshthemes/themes.zip

    echo ""
    echo "oh-my-posh updated! =)"
   fi
   ;;
*)
   echo "Usage: $0 {install|update}"
esac

exit 0