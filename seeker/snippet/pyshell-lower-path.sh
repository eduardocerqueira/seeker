#date: 2023-07-27T16:40:40Z
#url: https://api.github.com/gists/ec75af21058eb74a759739a4b3c4b8cc
#owner: https://api.github.com/users/volt-france

for f in $(ls -R ~/Downloads/programming); do fnew=$(python -c"from pathlib import Path; p = Path('$f'); print(str(p.resolve().parent).lower()) if '.' in p.name else None"); mv $f $fnew ; done 2> /dev/null
