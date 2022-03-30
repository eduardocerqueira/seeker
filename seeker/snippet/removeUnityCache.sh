#date: 2022-03-30T17:02:27Z
#url: https://api.github.com/gists/f06c6ef7a976ab33bb480b0949e384e9
#owner: https://api.github.com/users/cyber1496

#!/bin/zsh

# すべてのUnity中間ファイルを削除する
removeUnityCache() {
	prevDir=$(cd $(dirname $0); pwd)

	exts=(".csproj" ".sln")
	for (( i = 0; i < ${#exts[@]}; ++i ))
	do
		for file in `find . -name "*${exts[$i]}"`; do
			rm -f ${file}
		done
	done

	# 消されると困るものは適宜書き換え
	dirs=("Library" "Logs" "Temp" "obj" "build")
	for (( i = 0; i < ${#dirs[@]}; ++i ))
	do
		dir="${dirs[$i]}"
		if [ -e ${dir} ]; then
			rm -rf ${dir}
		fi
	done

	cd ${prevDir}
}

removeUnityCache