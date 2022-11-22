#date: 2022-11-22T17:03:55Z
#url: https://api.github.com/gists/73892c1fb7459e988d243462922e85cb
#owner: https://api.github.com/users/TrungThanhTran

def inference(gpt2_pipe, starting_text):
    try:
        response = gpt2_pipe(starting_text, max_length=(len(starting_text) + random.randint(60, 90)), num_return_sequences=4)
        response_list = []
        for x in response:
            resp = x['generated_text'].strip()
            if resp != starting_text and len(resp) > (len(starting_text) + 4) and resp.endswith((":", "-", "—")) is False:
                response_list.append(resp+'\n')

        response_end = "\n".join(response_list)
        response_end = re.sub('[^ ]+\.[^ ]+','', response_end)
        response_end = response_end.replace("<", "").replace(">", "")

        if response_end != "":
            return response_end
    except Exception as e:
        print('Exception in ps_inference = ', e)
        return starting_text