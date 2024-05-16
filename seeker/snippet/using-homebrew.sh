#date: 2024-05-16T16:51:34Z
#url: https://api.github.com/gists/2de7300048d74c367aadb587f8e81fee
#owner: https://api.github.com/users/janarthanancs

# using Homebrew

# Note: `brew install npm` has problems, as of 2010-12-30.
# hopefully it will eventually be good and happy.
# As of npm@0.2.13, however, this is an option

PREFIX=$(brew --prefix)

# take ownership
# this will also let homebrew work without using sudo
# please don't do this if you don't know what it does!
sudo mkdir -p $PREFIX/{share/man,bin,lib/node,include/node}
sudo chown -R $USER $PREFIX/{share/man,bin,lib/node,include/node}

brew install node

# now install npm
# prefix will default to $(brew --prefix)
curl https://www.npmjs.org/install.sh | sh
