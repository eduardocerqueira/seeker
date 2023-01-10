#date: 2023-01-10T16:54:21Z
#url: https://api.github.com/gists/3fcef47871c5c2bcc11e375bb1e14ffc
#owner: https://api.github.com/users/fjs138


sudo apt-get update &&
echo "\e[34mapt-get update updates the local package index with the latest changes made in the repositories"
echo the APT package index is essentially a database of available packages
echo the repositories are defined in the /etc/apt/sources.list file and in the /etc/apt/sources.list.d directory.
echo "\e[0m"
sudo apt-get dist-upgrade -y &&
echo "\e[34mapt-get dist-upgrade -y is used to install the newest versions of all packages
echo currently installed on the system from the sources enumerated in
echo /etc/apt/sources.list
echo dist-upgrade, in addition to performing the function of upgrade"
echo also intelligently handles changing dependencies with new versions of packages
echo "\e[0m"

sudo apt-get autoremove &&
echo "\e[34mapt-get autoremove is used to remove packages that were automatically installed to satisfy dependencies that are no longer needed"
echo "\e[0m"
sudo apt-get remove --purge $(deborphan) &&
echo "\e[34mapt-get remove --purge $(deborphan) finds packages that have no packages depending on them, that you did not explicitly install"
echo removes them
echo sudo apt-get -f install &&
echo sudo apt-get install -f &&
echo this does not work in-script, but works from prompt, not sure why yet
echo attempt to correct a system with broken dependencies in place.
echo "\e[0m"
sudo dpkg --configure -a
echo "\e[34msudo dpkg --configure -a completes interrupted package operations"