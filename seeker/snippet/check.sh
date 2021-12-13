#date: 2021-12-13T17:12:59Z
#url: https://api.github.com/gists/5d701751f3177f612faa21ca46b53f54
#owner: https://api.github.com/users/spy86

#!/bin/bash
echo "checking for log4j vulnerability...";
if [ "$(locate log4j|grep -v log4js)" ]; then
  echo "### maybe vulnerable, those files contain the name:";
  locate log4j|grep -v log4js;
fi;
if [ "$(dpkg -l|grep log4j|grep -v log4js)" ]; then
  echo "### maybe vulnerable, installed packages:";
  dpkg -l|grep log4j;
fi;
if [ "$(which java)" ]; then
  echo "java is installed, so note that Java applications often bundle their libraries inside jar/war/ear files, so there still could be log4j in such applications.";
fi;
echo "If you see no output above this line, you are safe. Otherwise check the listed files and packages.";