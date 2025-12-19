#date: 2025-12-19T16:57:25Z
#url: https://api.github.com/gists/d12621d5571bff13e520e861ad5758af
#owner: https://api.github.com/users/sgouda0412


####################
#  CLI cheatsheet  #
####################

# cheatsheet directory location:
CHEATDIR=~/.cheatsheet

# Use 'vicheat' to edit the cheatsheet
alias vicheat='vim $CHEATDIR/command_cheatsheet'

# Use 'cheat <word>' to search for <word> in the cheatsheet
cheat() {
    grep -iw --color=auto $1 $CHEATDIR/command_cheatsheet
}

# OPTIONAL: Use 'thecheat <word>' to have cheatsheet results delivered in color by Homestar Runner's The Cheat. 
# (Requires cowsay, lolcat, and thecheat.cow)
thecheat () {
    grep -iw $1 $CHEATDIR/command_cheatsheet | cowsay -f thecheat -n | lolcat
}

