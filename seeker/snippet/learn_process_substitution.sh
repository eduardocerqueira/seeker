#date: 2022-01-20T17:03:30Z
#url: https://api.github.com/gists/3570d9f067ae2ad03fecbbd997e05549
#owner: https://api.github.com/users/kis9a

## process substitution
# https://qiita.com/angel_p_57/items/fd1d9a10e2f4aba17669
# https://techblog.raccoon.ne.jp/archives/53726690.html
function process_substitution_example() {
  seq 10 | tee >(grep -v 1 > output)
}

function process_substitution_diff_example() {
  diff <(echo "apple\norange") <(echo "orange\nbanana")
}

function process_substitution_fd() {
  echo <(seq 10) # ファイルディスクリプタ
}

function process_substitution_ps_example() {
  diff -y <(ps ax) <(sleep 5; ps ax)
}