#date: 2023-03-03T17:06:07Z
#url: https://api.github.com/gists/38952fa2f789a47845484461ea8de966
#owner: https://api.github.com/users/dev-head

#!/usr/bin/env bash

set -e

#:-{Define Functions Ya'll}------------------------------------------------------------------------------->

function install_brew {

    # install xcode cli tools. (required dependency for homebrew and life in general)
    if [ ! $(pkgutil --pkgs=com.apple.pkg.CLTools_Executables) ]; then
        echo "++[install]::[xcode-cli]"
        #xcode-select --install
    fi

    if ! command -v brew &> /dev/null; then
        echo "++[install]::[homebrew]"
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install.sh)"
    fi

    echo "++[update]::[brew]"
    brew update
}

function install_packages {
    echo "++[install]::[packages]"
    PACKAGES=(
        ack
        autoconf
        automake
        awscli
        bash
        cowsay
        coreutils
        docker
        ffmpeg
        findutils
        fortune
        gettext
        gifsicle
        git
        git-flow
        # failing tls graphviz
        gpg-suite
        htop
        hub
        imagemagick
        jq
        libjpeg
        libmemcached
        lynx
        markdown
        mysql-client
        mydumper
        nmap
        npm
        pkg-config
        python
        python3
        pypy
        percona-toolkit
        rename
        screen
        ssh-copy-id
        terminal-notifier
        the_silver_searcher
        tmux
        travis
        tree
        vim
        wget
        zsh
        zoom
        minikube
    )

    brew install ${PACKAGES[@]}
    brew cleanup
}

function install_casks {
    echo "++[install]::[casks]"
    CASKS=(
        atom
        bbedit
        cyberduck
        docker
        dropbox
        flux
        google-chrome
        iterm2
        magicprefs
        skype
        slack
        spectacle
        spotify
        sublime-text
        sequel-pro
        textmate
        transmit
        tunnelblick
        vagrant
        virtualbox
        vlc
        vox
    )

    brew install --cask ${CASKS[@]}
}


function install_python_packages {
    echo "++[install]::[python packages]"
    PYTHON_PACKAGES=(
        boto
        ipython
        simplejson
        #simpleyaml
        csvfilter
        virtualenv
        virtualenvwrapper
    )
    sudo pip3 install ${PYTHON_PACKAGES[@]}
}

function install_terraform {
    echo "++[installing]::[terraform]"
    brew tap hashicorp/tap
    brew install hashicorp/tap/terraform

    echo "++[installing]::[chtf]"
    brew tap Yleisradio/terraforms
    brew install chtf
}

function configure_os {
    echo "++[configure]::[os]"

    # Reset perms in homebrew just in case you don't own it. 
    # @TODO: prompt for user if perm issue. 
    # sudo chmod -R g+rwx /usr/local/* 
    # sudo chown -R $(whoami):admin /usr/local/* 

    # Set fast key repeat rate
    defaults write NSGlobalDomain KeyRepeat -int 0

    # Require password as soon as screensaver or sleep mode starts
    defaults write com.apple.screensaver askForPassword -int 1
    defaults write com.apple.screensaver askForPasswordDelay -int 0

    # Show filename extensions by default
    defaults write NSGlobalDomain AppleShowAllExtensions -bool true

    # Enable tap-to-click
    defaults write com.apple.driver.AppleBluetoothMultitouch.trackpad Clicking -bool true
    defaults -currentHost write NSGlobalDomain com.apple.mouse.tapBehavior -int 1

    # Disable "natural" scroll
    defaults write NSGlobalDomain com.apple.swipescrolldirection -bool false

    # Clean out BS startups
    launchctl remove com.microsoft.teams.TeamsUpdaterDaemon
    launchctl remove com.microsoft.office.licensingV2.helper
    launchctl remove com.microsoft.OneDriveStandaloneUpdaterDaemon
    launchctl remove com.microsoft.OneDriveUpdaterDaemon
    launchctl remove com.microsoft.autoupdate.helper
    launchctl remove com.apple.systemstats.microstackshot_periodic

    ls -alih /Library/LaunchAgents
    ls -alih /Library/LaunchDaemons
    # log stream --predicate '(process == "WindowServer")' --debug

    rm /Library/LaunchAgents/com.microsoft.OneDriveStandaloneUpdater.plist
    rm /Library/LaunchAgents/com.microsoft.update.agent.plist
    rm /Library/LaunchDaemons/com.microsoft.OneDriveStandaloneUpdaterDaemon.plist
    rm /Library/LaunchDaemons/com.microsoft.OneDriveUpdaterDaemon.plist
    rm /Library/LaunchDaemons/com.microsoft.autoupdate.helper.plist
    rm /Library/LaunchDaemons/com.microsoft.office.licensingV2.helper.plist
    rm /Library/LaunchDaemons/com.microsoft.teams.TeamsUpdaterDaemon.plist
    rm /Library/LaunchDaemons/com.Amazon.WorkDocs.DriveUpdater.plist
    rm /Library/LaunchDaemons/com.amazonaws.acvc.helper.plist
    rm /Library/LaunchDaemons/com.docker.vmnetd.plist
    rm /Library/LaunchDaemons/net.tunnelblick.tunnelblick.tunnelblickd.plist
    rm /Library/LaunchDaemons/org.openvpn.client.plist
    rm /Library/LaunchDaemons/org.openvpn.helper.plist
    rm /Library/LaunchDaemons/org.virtualbox.startup.plist


}

function configure_dirs {
    echo "++[configure]::[dirs]"
    [[ ! -d ~/project ]] && mkdir ~/project
}

function configure_vagrant {
    echo "++[configure]::[vagrant]"
    vagrant plugin install vagrant-vbguest
}

function configure_zsh {
    echo "++[configure]::[zsh]"
    sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
    chsh -s /usr/local/bin/zsh
}

function configure {
    echo "+[start]::[configure]"
    configure_os
    configure_dirs
    configure_vagrant
    configure_zsh
    echo "+[end]::[configure]"
}

function install {
    echo "+[start]::[install]"
    install_brew
    install_packages
    install_casks
    install_terraform
    install_python_packages
    echo "+[end]::[install]"
}

function bootstrap {
    echo "[start]::[bootstrap]"
    install
    configure
    echo "[end]::[bootstrap]"
}

#:-{Execute how you do}------------------------------------------------------------------------------->
bootstrap
