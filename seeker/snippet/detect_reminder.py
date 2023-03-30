#date: 2023-03-30T16:45:31Z
#url: https://api.github.com/gists/67833d37f95f5f87401e39c12e969db7
#owner: https://api.github.com/users/alex000kim

import openai

def is_reminder_request(user_input):
    prompt = f"Does the following text contain a request to set a reminder?\n\n{user_input}\n\n(Yes/No):"

    openai_response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )

    answer = openai_response.choices[0].message.content.strip().lower()
    return answer == "yes"
