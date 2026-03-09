#date: 2026-03-09T17:43:29Z
#url: https://api.github.com/gists/606c45e33b4a138a4f9cfaa8f12ba348
#owner: https://api.github.com/users/n0nick

$ brew tap

sagie.maoz@sagie-farm1 ~
$ brew bundle -v --file=- <<-EOS
tap "viamrobotics/brews"
brew "canon"
EOS
Error: No available formula with the name "canon". Did you mean carton?
`brew bundle` failed! Failed to fetch canon

sagie.maoz@sagie-farm1 ~
$ brew bundle -v --file=- <<-EOS
tap "viamrobotics/brews"
brew "viamrobotics/brews/canon"
EOS
Tapping viamrobotics/brews
Installing viamrobotics/brews tap. It is not currently installed.
==> Tapping viamrobotics/brews
Cloning into '/home/linuxbrew/.linuxbrew/Homebrew/Library/Taps/viamrobotics/homebrew-brews'...
remote: Enumerating objects: 2873, done.
remote: Counting objects: 100% (1017/1017), done.
remote: Compressing objects: 100% (235/235), done.
remote: Total 2873 (delta 934), reused 823 (delta 782), pack-reused 1856 (from 3)
Receiving objects: 100% (2873/2873), 368.58 KiB | 4.14 MiB/s, done.
Resolving deltas: 100% (2112/2112), done.
Tapped 15 formulae (36 files, 594.7KB).
Installing viamrobotics/brews/canon
Installing canon formula. It is not currently installed.
==> Fetching downloads for: canon
✔︎ Bottle Manifest go (1.26.1)                        Downloaded    7.5KB/  7.5KB
✔︎ Bottle go (1.26.1)                                 Downloaded   66.4MB/ 66.4MB
✔︎ Formula canon (1.2.0)                              Verified     27.6KB/ 27.6KB
==> Installing canon from viamrobotics/brews
==> Installing viamrobotics/brews/canon dependency: go
==> Pouring go--1.26.1.x86_64_linux.bottle.tar.gz
🍺  /home/linuxbrew/.linuxbrew/Cellar/go/1.26.1: 14,930 files, 238.8MB
==> go build -o canon ./
^C
