#date: 2022-04-12T17:14:39Z
#url: https://api.github.com/gists/215d1de79786bec99814fee2b7ba37eb
#owner: https://api.github.com/users/mcsheehan

docker run -it -v mongodata:/data/db -p 27017:27017 --name mongodb -d mongo

# Then connect to the instance using compass - no credentials needed for running it locally.
connect on localhost:27017