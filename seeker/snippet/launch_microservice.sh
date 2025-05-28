#date: 2025-05-28T16:47:14Z
#url: https://api.github.com/gists/c0235b6eb1b7e61996bc2c8a4c421515
#owner: https://api.github.com/users/jaehoo

#!/bin/bash

export JAVA_HOME = "C:\Program Files\Java\jdk-17"
export PATH="$JAVA_HOME\bin" +":"+ $PATH

# Launch microservice
java -jar ./calculator-soap-oauth2-0.0.1-SNAPSHOT.jar

