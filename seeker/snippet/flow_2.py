#date: 2022-04-04T17:06:18Z
#url: https://api.github.com/gists/ac442c67c694fe1ca9f0fc4107ee6cac
#owner: https://api.github.com/users/achintya-7

with flow as f:    
 f.post(on="/index", inputs=movies, show_progress=True)     
 f.post(on="/", show_progress=True)
 f.cors = True
 f.block()