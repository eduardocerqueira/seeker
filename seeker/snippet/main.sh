#date: 2022-03-30T16:50:50Z
#url: https://api.github.com/gists/177f083f16ae83e82c9c61ccfabbbca3
#owner: https://api.github.com/users/b0rdjack

cob() {
  git checkout $1
  git fetch origin

  update_js="$(git diff --name-only origin/$1 package.json)"
  update_sql="$(git diff --name-only origin/$1 db/structure.sql)"
  update_gem="$(git diff --name-only origin/$1 Gemfile)"

  git pull

  if [ -n "${update_js}" ]; then
     yarn install --check-files
  fi

  if [ -n "${update_sql}" ]; then
    rails db:reset
  fi

  if [ -n "${update_gem}" ]; then
    bundle
  fi
}