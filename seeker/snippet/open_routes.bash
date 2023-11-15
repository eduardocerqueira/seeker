#date: 2023-11-15T16:57:30Z
#url: https://api.github.com/gists/14893f5b1348a435bd9f7cad262c15de
#owner: https://api.github.com/users/MichaelDimmitt

#!/bin/bash
# This code allows for pagination, 4 comments below say: "remove me to turn off pagination"
{ 
  paginationCursor=1; # remove me to turn off pagination
  paginationScalar=5; # remove me to turn off pagination
  
  baseURL="https://localhost"; 
  listOfRoutes=$(
    rails routes | 
    grep GET | 
    tail -n $(( paginationCursor * paginationScalar )) | # remove me to turn off pagination
    head -n $paginationScalar |                          # remove me to turn off pagination
    awk -F '  /' '{print $2}' | awk '{print $1}'
  ); 

  echo "$listOfRoutes" |
  xargs -I {} echo $baseURL/{} 
  # xargs -I {} open -a "Google Chrome" $baseURL/{} --args --auto-open-devtools-for-tabs; 
}

# ^ currently it prints all the constructed links:
# └-- uncomment the last line to have it open google chrome with the related links.