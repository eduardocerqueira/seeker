#date: 2024-08-14T18:24:28Z
#url: https://api.github.com/gists/4528722d4ad3b910d25adca208934323
#owner: https://api.github.com/users/wteuber

git ls-files | xargs wc -w 2> /dev/null | ruby -e "puts ARGF.map{_1.scan(/^\s*(\d+)/)[0][0].to_i}.inject(&:+)"