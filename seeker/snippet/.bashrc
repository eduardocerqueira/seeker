#date: 2025-04-21T16:54:56Z
#url: https://api.github.com/gists/db6fa34a149624b6739f89e002df4b25
#owner: https://api.github.com/users/pegeler

# Minimal .bashrc that calls other startup scripts
# DO NOT ALTER: make modifications/additions to
# files in ~/.profile.d/


# If not running interactively, don't do anything
case $- in
    *i*) ;;
      *) return;;
esac

shopt -s nullglob

function source_module () {
    test -f "$1" && echo "Sourcing $1" && . "$1"
}

export -f source_module

# Do the likely suspects such as bash_aliases, etc
source_module "$HOME/.bash_aliases"
# ... more to come ...

# Get the platform we're running on.
# To customize for OS, we'll source any shell file in
# ~/.profile.d/$platform
case "$(uname -s)" in
  Linux)
    platform=linux
    ;;
  CYGWIN*|MSYS*|MINGW*)
    platform=win
    ;;
  Darwin|*)
    platform=mac
    ;;
esac

if [ -d "$HOME/.profile.d" ]; then
  for i in "$HOME/.profile.d/"*.sh; do
    source_module "$i"
  done

  if [ -d "$HOME/.profile.d/$platform" ]; then
    echo "Loading $platform config files for: $(uname -s)"
    for i in "$HOME/.profile.d/$platform/"*.sh; do
      source_module "$i"
    done
  fi
  
  unset platform
  unset i
fi
