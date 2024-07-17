#date: 2024-07-17T16:39:19Z
#url: https://api.github.com/gists/ee052b966c950b6d040b0c7fa47633d6
#owner: https://api.github.com/users/mpo-vliz

#! /bin/bash

# endpoints 
ENDPOINTS=("https://id.cabi.org/PoolParty/sparql/cabt" "https://dbpedia.org/sparql")
# origins
ORIGINS=("https://query.linkeddatafragments.org/" "https://yasgui.org/proxy/" "https://anydomain.org/" "https://id.cabi.org/") 


#constantly reuse this sparql query statement for test
QRY="SELECT * WHERE {?s ?p ?o.} LIMIT 3"

function test_cors {
  ENDPOINT=$1         # URL of endpoint
  ORIGIN=$2           # URL of origin
  printf "\n--\nchecking endpoint: %-50s\n    from origin  : %-50s...\n" "${ENDPOINT}" "${ORIGIN:-«none»}"

  args=()                        # initialise array
  args+=("-L")                   # follow links
  #args+=("-v")                   # verbose output for headers 
  args+=("-o" "/dev/null")       # ignore the actual content
  args+=("-s" "-D" "-")          # hide progress (silent) and dump headers

  args+=("--url" "${ENDPOINT}")  # the endpoint to test 
  args+=("--data-urlencode" "query=$QRY")                  # sparql query to perform
  args+=("-H" "Accept: application/sparql-results+json")   # conneg for type response
  if [ ! -z "${ORIGIN}" ]; then  # only if an origin is provided
    args+=("-H" "Origin: ${ORIGIN}")    # we should pass it along
  fi

  resp_hdrs=$(curl "${args[@]}" 2>&1 ) 
  status=$(echo "$resp_hdrs"| head -n 1 | xargs) # xargs will strip whitespace effectively

  if [[ "${status}" =~ ^HTTP.*200$ ]]; then # succesful response starts with HTTP and ends with 200
    echo "   +++ RESPONSE OK    --> $status"
    if [ -z "${ORIGIN}" ]; then 
      echo "   and no origin to check"
    else
      echo "   checking match origin ${ORIGIN}"
      did_match_origin=$(echo "$resp_hdrs" | grep -i -P "^access-control-allow-origin: (\*|$ORIGIN)" | wc -l)
      if [ "$did_match_origin" -eq "1" ]; then
        echo  "      +++ OK, there was an origin-allow-match !"
      else
        echo  "      *** ERROR no origin-allow-match !"
      fi
    fi
  else
    echo "   *** ERROR RESPONSE --> $status"
  fi
}

echo "Testing CORS on combo of "
echo "endpoints ${ENDPOINTS[@]}"
echo "and origins ${ORIGINS[@]}" 
echo "------------------------------"

for e in ${ENDPOINTS[@]}; do 
  for o in ${ORIGINS[@]}; do
    test_cors $e $o
  done
  test_cors $e
done

echo "------------------------------"
echo "done"

