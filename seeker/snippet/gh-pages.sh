#date: 2021-12-23T17:08:20Z
#url: https://api.github.com/gists/146c1ec1ced831c0ca3b1f54972775b6
#owner: https://api.github.com/users/weibobo

# create gh-pages branch
git checkout --orphan gh-pages
git rm -rf .
touch README.md
git add README.md
git commit -m 'initial gh-pages commit'
git push origin gh-pages

# add gh-pages as submodule
git checkout master
git submodule add -b gh-pages git@github.com:skratchdot/MYPROJECT.git _site
git commit -m "added gh-pages as submodule"
git push origin master
git submodule init
