#date: 2024-12-24T17:02:52Z
#url: https://api.github.com/gists/a04845d424a011b739194d690607a4b3
#owner: https://api.github.com/users/Kelniit

#!/bin/bash

# Create Cloud Storage

gcloud storage buckets create gs://pemilu-storage

#

gcloud storage cp sample.jpg gs://pemilu-storage

#

gcloud storage cp -r gs://pemilu-storage/sample.jpg

#

gcloud storage cp gs://pemilu-storage/sample.jpg gs://pemilu-storage/images

#

gcloud storage ls gs://pemilu-storage

#

gcloud storage ls -l gs://pemilu-storage/sample.jpg