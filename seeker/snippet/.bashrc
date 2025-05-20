#date: 2025-05-20T17:02:45Z
#url: https://api.github.com/gists/1231568c8ce1a132806d1aac30127501
#owner: https://api.github.com/users/atakanargn

# Git Aliases with 'g' Prefix
alias g='git'
alias gcl='g clone "$@"'                      # gcl https://github.com/user/repo.git
alias gco='g commit "$@"'                     # gco -m "message"
alias gaa='g add .'
alias gst='g status'
alias gl='g log --oneline --graph --decorate --all'
alias gpl='g pull "$@"'                       # gpl origin main
alias gp='g push "$@"'                        # gp origin main
alias gcm='g checkout main'
alias gcb='g checkout -b "$@"'                # gcb new-branch
alias gch='g checkout "$@"'                   # gch existing-branch
alias gb='g branch'
alias gbd='g branch -d "$@"'                  # gbd branch-name
alias gr='g remote -v'
alias grs='g reset --soft HEAD~1'
alias grh='g reset --hard HEAD~1'
alias gam='g commit --amend --no-edit'
alias gcp='g cherry-pick "$@"'                # gcp commit_hash
alias gtag='g tag "$@"'                       # gtag v1.0.0
alias gpsup='g push --set-upstream origin $(g branch --show-current)'

# Stash commands
alias gss='g stash'
alias gsp='g stash pop'

# Rebase commands
alias grb='g rebase "$@"'                     # grb main
alias grbc='g rebase --continue'
alias grba='g rebase --abort'

# git starter
alias gnew='g init && gaa && gco -m "first commit" && g remote add origin "$1" && gp -u origin main'
