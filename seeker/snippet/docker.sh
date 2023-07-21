#date: 2023-07-21T16:53:51Z
#url: https://api.github.com/gists/b0c1d2ba6f1e69ea5676df87cd683c52
#owner: https://api.github.com/users/matteocastiglioni

#!/bin/bash

case $(uname | tr '[:upper:]' '[:lower:]') in
	linux*) # Linux
		if [[ $(lsb_release -d | awk -F"\t" '{print $2}' | awk -F " " '{print $1}' | tr '[:upper:]' '[:lower:]') == 'ubuntu' ]]; then
			function remove_dc_dkc_files {
				sudo rm -rf /var/lib/docker
				sudo rm -rf /var/lib/containerd
			}
			
			function aliases_status {
				echo The following aliases were $1:
				echo "dk => docker"
				echo "dkc => docker compose"
				echo "dkstart => sudo service docker start"
				echo "dkstop => sudo service docker stop"
			}
			
			echo "What do you want to do?"
			echo "1) Install Docker and Docker Compose"
			echo "2) Update Docker and Docker Compose"
			echo "3) Uninstall Docker and Docker Compose"
			echo "4) Remove Docker and Docker Compose images, containers, volumes, networks and customised configuration files"
			read -p ": " action
			
			if [[ $action == "1" || $action == "2" ]]; # install or update
			then
				# required by Laravel Sail
				sudo apt-get install libsecret-1-0 -y
				sudo apt-get install pass -y
				
				# install docker
				# https://docs.docker.com/engine/install/ubuntu/#install-using-the-repository		
				sudo apt-get update
				sudo apt-get install ca-certificates curl gnupg
				sudo install -m 0755 -d /etc/apt/keyrings
				curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
				sudo chmod a+r /etc/apt/keyrings/docker.gpg
				echo "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
				sudo apt-get update
				sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
				echo 'alias dk="docker"' >> ~/.bashrc # add dk alias

				# install docker compose
				# https://docs.docker.com/compose/install/linux/#install-using-the-repository
				sudo apt-get update
				sudo apt-get install docker-compose-plugin
				echo 'alias dkc="docker compose"' >> ~/.bashrc # add dkc alias

				# create start and stop alias
				echo 'alias dkstart="sudo service docker start"' >> ~/.bashrc # add dkstart alias
				echo 'alias dkstop="sudo service docker stop"' >> ~/.bashrc # add dkstop alias
				
				echo $'\nDONE!\n'
				
				if [[ $action == "1" ]]; # install
				then
					aliases_status 'created'
				fi
			elif [[ $action == "3" ]]; # uninstall
			then
				# uninstall docker
				# https://docs.docker.com/engine/install/ubuntu/#uninstall-docker-engine
				for pkg in docker.io docker-doc docker-compose podman-docker containerd runc;
					do sudo apt-get remove $pkg;
				done
				
				sudo apt-get purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras

				# uninstall docker compose
				# https://docs.docker.com/compose/install/uninstall/#uninstalling-the-docker-compose-cli-plugin
				sudo apt-get remove docker-compose-plugin
				rm $DOCKER_CONFIG/cli-plugins/docker-compose
				rm /usr/local/lib/docker/cli-plugins/docker-compose

				# Inspect the location of the Compose CLI plugin
				# docker info --format '{{range .ClientInfo.Plugins}}{{if eq .Name "compose"}}{{.Path}}{{end}}{{end}}'
				
				# remove alias
				sed -i '/alias dk="docker"/d' ~/.bashrc # remove dk alias
				sed -i '/alias dkc="docker compose"/d' ~/.bashrc # remove dkc alias
				sed -i '/alias dkstart="sudo service docker start"/d' ~/.bashrc # remove dkstart alias
				sed -i '/alias dkstop="sudo service docker stop"/d' ~/.bashrc # remove dkstop alias
				
				read -p $'\nDo you also want to remove images, containers, volumes, networks and customised configuration files (y/n)? ' answer
				if [[ $answer == "y" ]]; then
					remove_dc_dkc_files
				fi
				
				echo $'\nDONE!\n'
				
				if [[ $answer == "y" ]]; then
					echo "The following folders were removed:"
					echo "/var/lib/docker"
					echo $'/var/lib/containerd\n'
				fi
				
				aliases_status 'removed'
			elif [[ $action == "4" ]]; # remove files
			then
				read -p $'\nThe cancellation is permanent and cannot be undone, are you sure you want to proceed (y/n)? ' confirm
				if [[ $confirm == "y" ]]; then
					remove_dc_dkc_files
				
					echo $'\nDONE!\n'
					echo "The following folders were removed:"
					echo "/var/lib/docker"
					echo "/var/lib/containerd"
				fi
			else
				echo $'\nChoice doesn\'t exist!'
			fi

			exec bash # reload bash
			
			# OPTIONAL: https://docs.docker.com/engine/install/linux-postinstall
		fi
    ;;
    darwin*) # OSX
        echo "OSX OS"
    ;;
    *)
        echo "Unknown OS"
    ;;
esac
