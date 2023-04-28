#date: 2023-04-28T16:53:58Z
#url: https://api.github.com/gists/457b1a214483037ff64738a3c72a0359
#owner: https://api.github.com/users/PubliusAu

def call_openai_api(context, prompt=None):
    if prompt:
        context.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
        model=model,  # Replace with the appropriate model name for GPT-4
        messages=context,
        temperature=0,
    )
    ai_message = response["choices"][0]["message"]["content"].strip()
    context.append({"role": "assistant", "content": ai_message})
    return ai_message