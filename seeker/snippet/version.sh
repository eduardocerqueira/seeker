#date: 2023-07-26T17:06:00Z
#url: https://api.github.com/gists/91f33eeced50f6aa93701e91a2e93a6b
#owner: https://api.github.com/users/kotwanikunal


          TAG="2.9.0"
          CURRENT_VERSION_ARRAY=($(echo "$TAG" | tr . '\n'))
          BASE=$(IFS=. ; echo "${CURRENT_VERSION_ARRAY[*]:0:2}")
          BASE_X=$(IFS=. ; echo "${CURRENT_VERSION_ARRAY[*]:0:1}.x")
          CURRENT_VERSION=$(IFS=. ; echo "${CURRENT_VERSION_ARRAY[*]:0:3}")
          CURRENT_VERSION_UNDERSCORE=$(IFS=_ ; echo "V_${CURRENT_VERSION_ARRAY[*]:0:3}")
          CURRENT_VERSION_ARRAY[2]=$((CURRENT_VERSION_ARRAY[2]+1))
          NEXT_VERSION=$(IFS=. ; echo "${CURRENT_VERSION_ARRAY[*]:0:3}")
          NEXT_VERSION_UNDERSCORE=$(IFS=_ ; echo "V_${CURRENT_VERSION_ARRAY[*]:0:3}")
          if [[ ${#CURRENT_VERSION_ARRAY[2]} -gt 1 ]]; then            
            NEXT_VERSION_ID="${CURRENT_VERSION_ARRAY[0]:0:3}0${CURRENT_VERSION_ARRAY[1]:0:3}${CURRENT_VERSION_ARRAY[2]:0:3}99"
          else
            NEXT_VERSION_ID=$(IFS=0 ; echo "${CURRENT_VERSION_ARRAY[*]:0:3}99")
          fi
          echo "TAG=$TAG"
          echo "BASE=$BASE"
          echo "BASE_X=$BASE_X"
          echo "CURRENT_VERSION=$CURRENT_VERSION"
          echo "CURRENT_VERSION_UNDERSCORE=$CURRENT_VERSION_UNDERSCORE"
          echo "NEXT_VERSION=$NEXT_VERSION"
          echo "NEXT_VERSION_UNDERSCORE=$NEXT_VERSION_UNDERSCORE"
          echo "NEXT_VERSION_ID=$NEXT_VERSION_ID"
          echo Incrementing $CURRENT_VERSION to $NEXT_VERSION
          echo "  - \"$CURRENT_VERSION\"" >> .ci/bwcVersions
          sed -i "s/opensearch        = $CURRENT_VERSION/opensearch        = $NEXT_VERSION/g" buildSrc/version.properties
          echo Adding $NEXT_VERSION_UNDERSCORE after $CURRENT_VERSION_UNDERSCORE
          sed -i "s/public static final Version $CURRENT_VERSION_UNDERSCORE = new Version(\([[:digit:]]\+\)\(.*\));/\0\n    public static final Version $NEXT_VERSION_UNDERSCORE = new Version($NEXT_VERSION_ID\2);/g" libs/core/src/main/java/org/opensearch/Version.java
          sed -i "s/CURRENT = $CURRENT_VERSION_UNDERSCORE;/CURRENT = $NEXT_VERSION_UNDERSCORE;/g" libs/core/src/main/java/org/opensearch/Version.java