#date: 2022-11-23T16:55:26Z
#url: https://api.github.com/gists/ea5da27b4c5c2a5db9eb69ca76249d49
#owner: https://api.github.com/users/dthoma8

FROM postgres:10.5

COPY /postgres-data . # CONTAINS INIT DATA

EXPOSE 5432