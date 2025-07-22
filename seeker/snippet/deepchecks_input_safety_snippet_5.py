#date: 2025-07-22T16:51:07Z
#url: https://api.github.com/gists/0ef2ee8d95fe1e603cf004aad2395461
#owner: https://api.github.com/users/MichalOleszak

model_names = [
    "ChatGPT 4.5 Preview",
    "Claude 3.7 Sonnet",
    "Llama 3.3 70B Instruct Turbo",
]

prompt_template = prompts[3]
for model_name in model_names:
    prompt = prompt_template.replace("{{MODEL_NAME}}", model_name)
    input_safety = score_input_safety(prompt, dc_client)
    print(f"""
        prompt_id: 3
        model_name: {model_name}
        input_safety: {input_safety}
    """)