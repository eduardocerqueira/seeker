#date: 2022-01-20T17:05:06Z
#url: https://api.github.com/gists/db33c4e8f10b92509fe0702543a536c1
#owner: https://api.github.com/users/kis9a

function p_arr_merge() {
  arr1=(1 2 3)
  arr2=(4 5 6)
  dst=("${arr1[@]}" "${arr2[@]}")
  echo "${dst[@]}"
}

function p_arr_length() {
  arr=(apple banana 1234 5678 candy)
  echo "${#arr[*]}"
}

function p_obj() {
  declare -A cost
  cost["apple"]=300
  cost["banana"]=100
  cost["candy"]=200
  echo ${cost["banana"]}"yen"
}

function p_string_line_to_arr_tr() {
  var="string1,string2,string3"
  arr=(`echo $var | tr ',' ' '`)
  echo "arr: ${arr[@]}"
  echo "leng: ${#arr[@]}"
  for a in $arr; do
    echo "---"
    echo $a
  done
}

function p_string_line_to_arr_read() {
  var="string1 string2 string3"
  if [ -n "$ZSH_VERSION" ]; then
    declare -a arr
    read -A arr <<< $var
  else
    read -a arr <<< $var
  fi
  echo "arr: ${arr[@]}"
  echo "leng: ${#arr[@]}"
  for a in $arr; do
    echo "---"
    echo $a
  done
}