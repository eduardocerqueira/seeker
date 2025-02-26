#date: 2025-02-26T17:04:10Z
#url: https://api.github.com/gists/7f6d2ac06e4f3cededa1435a287872a1
#owner: https://api.github.com/users/Carleux


# Fix windows path for linux
function clean_user_path {
    current_path=$(echo $PATH)
    current_path="$current_path"
    cleaned_paths=""
    org_ifs=$IFS
    IFS=$':'
    paths=($current_path)
    counted_paths="${#paths[@]}"
    path_counter=0
    echo "counted_paths: $counted_paths"
    echo "Adding single quotes to fix space on Windows directories"
    echo
    for CURRENT in "${paths[@]}"; do
      counter=0
      IFS='/' read -ra PATH_PART <<< "$CURRENT"
      for i in "${PATH_PART[@]}"; do  
        grepped=$( echo $i | grep -P '\s')
        if [[ -n $grepped ]]; then
          quoted="'"$grepped"'"
          PATH_PART[$counter]=$quoted
        fi
        counter=$((counter+1))
      done
      updated_path=$(IFS=/ ; echo "${PATH_PART[*]}")
      if [ $path_counter -eq 0 ]; then
        cleaned_paths="$updated_path"
      else
        cleaned_paths="$cleaned_paths:$updated_path"
      fi
      path_counter=$((path_counter+1)) 
    done
    
    IFS=$org_ifs
    export PATH=$cleaned_paths
    echo $PATH
}

clean_user_path