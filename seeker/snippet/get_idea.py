#date: 2022-11-22T17:09:20Z
#url: https://api.github.com/gists/0a36f9d51712e9f6a0b671e36763892d
#owner: https://api.github.com/users/TrungThanhTran

""" If there is no input text, we choose from ideas.txt """
    with open("ideas.txt", "r") as f:
        line = f.readlines()
    
    if starting_text == "":
        starting_text: str = line[random.randrange(0, len(line))].replace("\n", "").lower().capitalize()
        starting_text: str = re.sub(r"[,:\-–.!;?_]", '', starting_text)

    gen_prompt = inference(ps_gpt2, starting_text, line)