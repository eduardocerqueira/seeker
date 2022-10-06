#date: 2022-10-06T17:19:05Z
#url: https://api.github.com/gists/69d385a039cd8258c642b377f6e55eb4
#owner: https://api.github.com/users/remarkablemark

brew install php@7.4
brew link --force --overwrite php@7.4
brew services start php@7.4
echo 'export PATH="/opt/homebrew/opt/php@7.4/bin:$PATH"' >> ~/.zshrc # or ~/.bashrc
echo 'export PATH="/opt/homebrew/opt/php@7.4/sbin:$PATH"' >> ~/.zshrc # or ~/.bashrc
