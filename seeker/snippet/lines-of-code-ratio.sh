#date: 2024-01-16T17:09:23Z
#url: https://api.github.com/gists/27fa22e1413df4ff4c1ff8aa6cd71a85
#owner: https://api.github.com/users/BlurryRoots

#!/bin/bash
# Copyright (c) 2023-∞ blurryroots innovation qanat OÜ

count-lines-of-code-package () {
	# Counts only source files in main package.

	local files=$(find ./src/piper_whistle -iname "*.py" -type f)
	local count=$(for f in ${files}; do wc -l $f; done \
		| awk '{ print $1 }' \
		| awk '{s+=$1} END {printf "%.0f", s}')

	echo ${count}

	return 0
}

count-lines-of-code-tooling () {
	# Collects all source files relevant for tooling, building and configuring.

	local files=$(find . \
		-iname "*.py" -type f \
		-not -path "*/src/piper_whistle/*" \
		-not -path "*/build/lib/*" \
		-o -iname "*.sh" -o -iname "makefile" \
		-o -iname "pip.conf" -o -iname "*.cfg" \
		-o -iname "*.yaml" -o -iname "*.yml")
	local count=$(for f in ${files}; do wc -l $f; done \
		| awk '{ print $1 }' \
		| awk '{s+=$1} END {printf "%.0f", s}')

	echo ${count}

	return 0
}

main () {
	# Investigate the codebase and create a ratio of 'lines of code in production' to 'lines of code to produce'

	local root_path="${1}"
	local package_lines=0
	local tool_lines=0
	local ratio=0

	pushd "${root_path}" > /dev/null
		package_lines=$(count-lines-of-code-package)
		tool_lines=$(count-lines-of-code-tooling)
	popd > /dev/null
	ratio=$(echo "${tool_lines} / ${package_lines}.0" | bc -l)

	echo "package: ${package_lines}"
	echo "tooling: ${tool_lines}"
	echo "ratio: ${ratio:0:6}"

	return 0
}

main $*
