#date: 2023-03-06T16:58:11Z
#url: https://api.github.com/gists/c16f3c6118b347f4644f56bd5b58c4a5
#owner: https://api.github.com/users/james-coder

#!/usr/bin/python3
#
# This code is licensed under the MIT License found here: https://mit-license.org/

import openai
import sys

# To get this key: "**********"://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key
openai.api_key = "YOUR_API_GOES_HERE"

def main(argv):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt="What day of the week is today? Give the answer for Pacific time. Your answer should only be one word, the day of the week.",
    )
    try:
        say = response.choices[0].text.strip()
    except:
        print(f"Bad response: {response}")
        raise
    print(say)

if __name__ == "__main__":
    sys.exit(main(sys.argv))
