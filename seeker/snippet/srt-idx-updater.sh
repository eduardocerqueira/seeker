#date: 2023-12-08T16:45:41Z
#url: https://api.github.com/gists/ed0c17e94b0784ce4073a013804e8f18
#owner: https://api.github.com/users/xdel

#/bin/sh
# note: the script is unable to distinguish between bare number being found
# in closed captions and the index itself, may be previous line's 
# cr/crlf shoulf be handled in a proper way
COUNT=$1
FILENAME=$2
sed -i -r 's/^([0-9]+)\r$/echo "$((\1+'$COUNT'))"/ge' $FILENAME