#date: 2023-09-11T17:00:26Z
#url: https://api.github.com/gists/79fb387d1fcb9b4c84c0f2f6c45e63c8
#owner: https://api.github.com/users/joshreini1

@retry(stop=stop_after_attempt(10), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_tru_query_engine(prompt):
    return tru_query_engine.query(prompt)
for prompt in test_prompts:
    call_tru_query_engine(prompt)