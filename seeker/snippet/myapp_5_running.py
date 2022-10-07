#date: 2022-10-07T17:14:37Z
#url: https://api.github.com/gists/213f4bcf07b68bb4d2a02df8f019a30a
#owner: https://api.github.com/users/jdegene

# 5 run the darn thing
if __name__ == "__main__":
    
    # this will run the app on your local machine: basically dash will run 
    # its own dev server on localhost: visit 127.0.0.1:8050 in your browser to
    # see your app
    app.run_server(port=8050, debug=True)

    #HOWEVER, if you are deploying it later, set your host to "0.0.0.0",
    # because dash will not be responsible for managing the server itself
    #app.run_server(host="0.0.0.0", port=8050)