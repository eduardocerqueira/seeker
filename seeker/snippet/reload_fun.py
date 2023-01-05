#date: 2023-01-05T17:09:53Z
#url: https://api.github.com/gists/4e31a81073ae2ff3e6a0549ccf496d67
#owner: https://api.github.com/users/isConic

if __name__ == "__main__":
    import uvicorn
    PORT = 9999
    
    ##### if this file is called "server.py"  then module name is server. 
    ##### if the app name is called "app" the the app name is app
    
    
    
    
    uvicorn.run("server:app",
                 port = PORT,
                reload = True)