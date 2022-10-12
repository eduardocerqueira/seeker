#date: 2022-10-12T17:04:24Z
#url: https://api.github.com/gists/7113baadb13208d91f4271a28debc58b
#owner: https://api.github.com/users/MahouShoujoMivutilde

#!/usr/bin/env bash

read -r -d '' copyqjs << EOM
for (var i = 0; i < size(); i++) {
  var lines = str(read(i)).split(/[\r\n]+/);
  var line = "";

  if (lines.length > 1) {
    line = lines[0] + " (+" + str(lines.length - 1) + " more lines)";
  } else {
    line = lines[0];
  }

  if (line == "") {
      line = '<probably image>';
  }

  print(i + " " + line + "\n");
}
EOM


pick="$(echo "$copyqjs" | copyq eval - | dmenu -i -l 20 | awk '{print $1}')"


if [[ "$pick" != "" ]]; then
    copyq select "$pick"
fi
