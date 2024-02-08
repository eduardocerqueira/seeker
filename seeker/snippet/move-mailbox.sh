#date: 2024-02-08T17:00:54Z
#url: https://api.github.com/gists/2a6b3f9330367d3bb43ceeb412bb7451
#owner: https://api.github.com/users/pintofbeer

#!/bin/bash

docker run --rm gilleslamiral/imapsync imapsync \
   --host1 source-emailsrvr.com --ssl1 --user1 mymail@mydomain.com --password1 'sourcepass' \
   --host2 destination-emailsrvr.com --ssl2 --user2 mymail@mydomain.com --password2 'destpass'