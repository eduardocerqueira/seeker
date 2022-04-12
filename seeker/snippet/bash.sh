#date: 2022-04-12T17:08:40Z
#url: https://api.github.com/gists/0af07d61eb1c096dc734d03b10a97f31
#owner: https://api.github.com/users/matthias-p-nowak

checkLogSize() {
  S=$(stat -L -c%s ${LFN} )
  # echo "size is ${S}"
  if [ ${S} -ge 16777216 ]
  then
    ln -sfv "${LFN}-$(date +%F_%H-%M)" ${LFN}
  fi
  exec >>${LFN} 2>&1
}