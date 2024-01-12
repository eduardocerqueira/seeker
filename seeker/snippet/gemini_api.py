#date: 2024-01-12T16:58:31Z
#url: https://api.github.com/gists/22a9475797f983d6aa2fdd6c55d0ff79
#owner: https://api.github.com/users/madaan

# MWE for using the Gemini api. The code has been tested with v0.3.2. 

import google.generativeai as genai
import random
import time

assert genai.__version__ == '0.3.2'

genai.configure(api_key="YOUR_KEY_HERE!")

model = genai.GenerativeModel('gemini-pro')




# from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple = Exception,  # Changed to catch all exceptions by default
):
    """Retry a function with exponential backoff."""
    def wrapper(*args, **kwargs):
        num_retries = 0
        delay = initial_delay

        while True:
            try:
                return func(*args, **kwargs)
            except errors as e:
                num_retries += 1
                if num_retries > max_retries:
                    print(f"Maximum number of retries ({max_retries}) exceeded.")
                    return None  # Changed to return None instead of raising an exception

                delay *= exponential_base * (1 + jitter * random.random())
                print(f"Retrying in {delay:.2f} seconds due to error: {e}")
                time.sleep(delay)
    return wrapper


  
@retry_with_exponential_backoff
 "**********"d "**********"e "**********"f "**********"  "**********"c "**********"a "**********"l "**********"l "**********"_ "**********"g "**********"e "**********"m "**********"i "**********"n "**********"i "**********"_ "**********"a "**********"p "**********"i "**********"( "**********"p "**********"r "**********"o "**********"m "**********"p "**********"t "**********", "**********"  "**********"m "**********"a "**********"x "**********"_ "**********"t "**********"o "**********"k "**********"e "**********"n "**********"s "**********"= "**********"3 "**********"0 "**********"0 "**********", "**********"  "**********"t "**********"e "**********"m "**********"p "**********"e "**********"r "**********"a "**********"t "**********"u "**********"r "**********"e "**********"= "**********"0 "**********". "**********"0 "**********") "**********": "**********"
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
        candidate_count=1,
        max_output_tokens= "**********"
        temperature=temperature)
    )
    return response.text

  

