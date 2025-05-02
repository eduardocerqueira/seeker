#date: 2025-05-02T16:42:45Z
#url: https://api.github.com/gists/1e5e3ca72ab7bddb49f5d18b7972f16a
#owner: https://api.github.com/users/jcttrll

#!/usr/bin/env bash

randomString() {
	local LC_ALL=C IFS=
	local i byte word string
	local -ar alphabet=(
		A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
		a b c d e f g h i j k l m n o p q r s t u v w x y z
		0 1 2 3 4 5 6 7 8 9 - _
	)

	{
		read -r -d '' -n1 byte && printf -v byte '%d' "'$byte"
		string=${alphabet[byte >> 3]}

		for ((i = 0; i < 5; i++)); do
			read -r -d '' -n1 byte && printf -v word '%d' "'$byte"

			read -r -d '' -n1 byte && printf -v byte '%d' "'$byte"
			word=$(( (word << 8) | byte ))

			read -r -d '' -n1 byte && printf -v byte '%d' "'$byte"
			word=$(( (word << 8) | byte ))

			string+=${alphabet[word >> 18]}
			string+=${alphabet[(word >> 12) & 0x3f]}
			string+=${alphabet[(word >> 6) & 0x3f]}
			string+=${alphabet[word & 0x3f]}
		done
	} </dev/urandom

	echo "$string"
}

randomString
