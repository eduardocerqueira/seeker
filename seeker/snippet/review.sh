#date: 2023-11-09T16:38:50Z
#url: https://api.github.com/gists/c99bf132abde691533f117ef1dd20a85
#owner: https://api.github.com/users/ericboehs

review_pr() {
  pr_id=$1
  gh pr view $pr_id --comments; git fetch -q; out_of_date=$(git rev-list --left-right --count origin/$GIT_MASTER_BRANCH...$(echo "origin/$(gh pr view $pr_id --json headRefName | jq .headRefName | tr -d '\"')") | awk '{print $1}'); [[ $out_of_date -gt 20 ]] && echo "‼️  Branch is $out_of_date commits out of date with $GIT_MASTER_BRANCH."; gh pr checks $pr_id; gh pr diff $pr_id; echo -n "[approve] or request-changes? "; read review; gh pr review $pr_id --${review:-approve}
}

review_prs() {
  pr_id=$(GH_FORCE_TTY=100 gh pr list --limit 200 --search "is:pr is:open draft:false NOT WIP in:title review-requested:@me review:required -label:Lighthouse label:console-services-review" | fzf --ansi --preview 'GH_FORCE_TTY=100 gh pr view {1}' --preview-window down --header-lines 3 | awk '{print $1}' | tr -d '#');
  review_pr $pr_id
}