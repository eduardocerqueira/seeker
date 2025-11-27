#date: 2025-11-27T17:13:52Z
#url: https://api.github.com/gists/53b57751500e82c368506a0fd189b902
#owner: https://api.github.com/users/bramanda48

#!/bin/sh
set -e
# Docker Engine for Linux uninstallation script.
#
# This script is intended as a convenient way to completely remove Docker Engine,
# Docker CLI, containerd, and related packages from your system.
#
# WARNING: This script will:
# - Remove all Docker packages (docker-ce, docker-ce-cli, containerd.io, etc.)
# - Optionally remove Docker images, containers, volumes, and custom configs
# - Remove Docker's package repositories from your system
#
# Usage
# ==============================================================================
#
# To uninstall Docker and remove all data:
#
# 1. download the script
#
#   $ curl -fsSL https://your-url/uninstall-docker.sh -o uninstall-docker.sh
#
# 2. verify the script's content
#
#   $ cat uninstall-docker.sh
#
# 3. run the script with --dry-run to verify the steps it executes
#
#   $ sh uninstall-docker.sh --dry-run
#
# 4. run the script either as root, or using sudo to perform the uninstallation.
#
#   $ sudo sh uninstall-docker.sh
#
# Command-line options
# ==============================================================================
#
# --purge-data
#
# Use the --purge-data option to remove all Docker data including images,
# containers, volumes, and custom configurations:
#
#   $ sudo sh uninstall-docker.sh --purge-data
#
# --keep-repo
#
# Use the --keep-repo option to keep Docker's package repositories configured
# on your system (useful if you plan to reinstall later):
#
#   $ sudo sh uninstall-docker.sh --keep-repo
#
# ==============================================================================

DRY_RUN=${DRY_RUN:-}
PURGE_DATA=0
KEEP_REPO=0

while [ $# -gt 0 ]; do
	case "$1" in
		--dry-run)
			DRY_RUN=1
			;;
		--purge-data)
			PURGE_DATA=1
			;;
		--keep-repo)
			KEEP_REPO=1
			;;
		--*)
			echo "Illegal option $1"
			;;
	esac
	shift $(( $# > 0 ? 1 : 0 ))
done

command_exists() {
	command -v "$@" > /dev/null 2>&1
}

is_dry_run() {
	if [ -z "$DRY_RUN" ]; then
		return 1
	else
		return 0
	fi
}

get_distribution() {
	lsb_dist=""
	if [ -r /etc/os-release ]; then
		lsb_dist="$(. /etc/os-release && echo "$ID")"
	fi
	echo "$lsb_dist"
}

do_uninstall() {
	echo "# Executing docker uninstall script"

	if ! command_exists docker && ! command_exists dockerd; then
		echo "Docker does not appear to be installed on this system."
		echo "Nothing to uninstall."
		exit 0
	fi

	user="$(id -un 2>/dev/null || true)"

	sh_c='sh -c'
	if [ "$user" != 'root' ]; then
		if command_exists sudo; then
			sh_c='sudo -E sh -c'
		elif command_exists su; then
			sh_c='su -c'
		else
			cat >&2 <<-'EOF'
			Error: this uninstaller needs the ability to run commands as root.
			We are unable to find either "sudo" or "su" available to make this happen.
			EOF
			exit 1
		fi
	fi

	if is_dry_run; then
		sh_c="echo"
	fi

	# Detect distribution
	lsb_dist=$( get_distribution )
	lsb_dist="$(echo "$lsb_dist" | tr '[:upper:]' '[:lower:]')"

	echo
	echo "=========================================================================="
	echo "WARNING: This will remove Docker from your system."
	if [ "$PURGE_DATA" = "1" ]; then
		echo "WARNING: This will also DELETE all Docker images, containers, volumes,"
		echo "         and custom configurations (--purge-data flag detected)."
	else
		echo "Note: Docker data (images, containers, volumes) will be preserved."
		echo "      Use --purge-data flag to remove all Docker data."
	fi
	echo "=========================================================================="
	echo

	if ! is_dry_run; then
		printf "Press Ctrl+C now to abort, or wait 10 seconds to continue..."
		echo
		sleep 10
	fi

	# Stop Docker service if running
	echo "Stopping Docker service..."
	if command_exists systemctl; then
		(
			if ! is_dry_run; then
				set -x
			fi
			$sh_c 'systemctl stop docker.service || true'
			$sh_c 'systemctl stop docker.socket || true'
			$sh_c 'systemctl stop containerd.service || true'
		)
	fi

	# Uninstall based on distribution
	case "$lsb_dist" in
		ubuntu|debian|raspbian)
			echo "Removing Docker packages (Debian/Ubuntu)..."
			(
				if ! is_dry_run; then
					set -x
				fi
				$sh_c 'apt-get -y -qq remove docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras docker-model-plugin || true'
				if [ "$PURGE_DATA" = "1" ]; then
					$sh_c 'apt-get -y -qq purge docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras docker-model-plugin || true'
				fi
				$sh_c 'apt-get -y -qq autoremove'
			)

			if [ "$KEEP_REPO" != "1" ]; then
				echo "Removing Docker repository..."
				(
					if ! is_dry_run; then
						set -x
					fi
					$sh_c 'rm -f /etc/apt/sources.list.d/docker.list'
					$sh_c 'rm -f /etc/apt/keyrings/docker.asc'
					$sh_c 'apt-get -qq update >/dev/null'
				)
			fi
			;;

		centos|fedora|rhel)
			echo "Removing Docker packages (CentOS/RHEL/Fedora)..."
			pkg_manager="yum"
			if command_exists dnf; then
				pkg_manager="dnf"
			fi
			(
				if ! is_dry_run; then
					set -x
				fi
				$sh_c "$pkg_manager -y -q remove docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin docker-ce-rootless-extras docker-model-plugin || true"
			)

			if [ "$KEEP_REPO" != "1" ]; then
				echo "Removing Docker repository..."
				(
					if ! is_dry_run; then
						set -x
					fi
					$sh_c 'rm -f /etc/yum.repos.d/docker-ce.repo'
					$sh_c 'rm -f /etc/yum.repos.d/docker-ce-staging.repo'
					if command_exists dnf; then
						$sh_c 'dnf clean all'
					else
						$sh_c 'yum clean all'
					fi
				)
			fi
			;;

		*)
			echo
			echo "ERROR: Unsupported distribution '$lsb_dist'"
			echo "Please manually uninstall Docker for this distribution."
			echo
			exit 1
			;;
	esac

	# Remove Docker data if requested
	if [ "$PURGE_DATA" = "1" ]; then
		echo "Removing Docker data directories..."
		(
			if ! is_dry_run; then
				set -x
			fi
			$sh_c 'rm -rf /var/lib/docker'
			$sh_c 'rm -rf /var/lib/containerd'
			$sh_c 'rm -rf /etc/docker'
			$sh_c 'rm -rf /var/run/docker.sock'
			$sh_c 'rm -rf ~/.docker'
		)
	else
		echo
		echo "Docker data preserved at:"
		echo "  - /var/lib/docker (images, containers, volumes)"
		echo "  - /etc/docker (configuration)"
		echo
		echo "To remove this data, run:"
		echo "  sudo rm -rf /var/lib/docker /var/lib/containerd /etc/docker"
		echo
	fi

	# Remove Docker group
	if getent group docker >/dev/null 2>&1; then
		echo "Removing docker group..."
		(
			if ! is_dry_run; then
				set -x
			fi
			$sh_c 'groupdel docker || true'
		)
	fi

	echo
	echo "=========================================================================="
	echo "Docker has been successfully uninstalled from your system."
	if [ "$PURGE_DATA" = "1" ]; then
		echo "All Docker data has been removed."
	else
		echo "Docker data has been preserved (use --purge-data to remove)."
	fi
	if [ "$KEEP_REPO" = "1" ]; then
		echo "Docker repositories have been kept on your system."
	fi
	echo "=========================================================================="
	echo
}

# wrapped up in a function so that we have some protection against only getting
# half the file during "curl | sh"
do_uninstall