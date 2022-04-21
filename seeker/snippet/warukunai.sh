#date: 2022-04-21T17:07:33Z
#url: https://api.github.com/gists/9c40c1817ed15a6afef50315818e3c25
#owner: https://api.github.com/users/LiviaMedeiros

#!/bin/sh
BASEDIR="$(realpath "${1-$(pwd)}")"
PREFIX="test/parallel/test-"
WARUKUNAI="warukunai"
FILES=$(git -C "${BASEDIR}" diff --name-only master | grep "^${PREFIX}" | sed "s#^${PREFIX}##")
for FILE in ${FILES}
do
	git -C "${BASEDIR}" show master:"${PREFIX}${FILE}" > "${BASEDIR}/${PREFIX}${WARUKUNAI}-${FILE}"
done
"${BASEDIR}/tools/test.py" --report "${WARUKUNAI}"
for FILE in ${FILES}
do
	rm -f "${BASEDIR}/${PREFIX}${WARUKUNAI}-${FILE}"
done
