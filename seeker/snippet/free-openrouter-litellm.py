#date: 2026-02-10T17:55:04Z
#url: https://api.github.com/gists/8da7ef1f4de590cb81f90f92d4bb2d01
#owner: https://api.github.com/users/RajuDasa

from litellm import completion
from google.colab import userdata  #colab snippet, use .env

key = userdata.get('OPENROUTER_API_KEY') #or store it in os.environ[]

result = completion(model="openrouter/openrouter/free", 
                    messages=[{'role':'user', 'content':'what are your capabilities - in short'}], 
                    api_key=key)
print(result)