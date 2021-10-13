#date: 2021-10-13T16:51:25Z
#url: https://api.github.com/gists/a293117c866e30326cf37928bcb7f7b2
#owner: https://api.github.com/users/firedynasty

function browse() {
  hello_var=$(echo -n `pbpaste`)
  # set the clipboard to a variable
  char=":"
  # char in case there are multiple links 
  hello_var_2=$(awk -F"${char}" '{print NF-1}' <<< "${hello_var}")
  # checking for mulitple https: links
  if [ "$hello_var_2" -gt 1 ]; then
    # if there are multiple links then return this statement
    echo 'set str1'
    echo 'const myArr = str1.split(" ");'
    echo 'for (var i = 0; i < myArr.length; i++) {'
    echo "      window.open(myArr[i], '_blank')"
    echo '}'
  else
    # otherwise open the link in google chrome
    open -a 'google chrome' $hello_var

  fi

}

# to use this command
# copy a link (with https)
# type browse
# enter and done!