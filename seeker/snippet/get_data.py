#date: 2021-09-16T16:52:52Z
#url: https://api.github.com/gists/c6ebe744564050e72882bcc2754ad76b
#owner: https://api.github.com/users/opalczynski

def get_data(data_path="./data.yml"):
    with open(data_path, "r") as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    # we need to process the data a bit to match the HTML template;
    for invoice in data["invoices"]:
        # here we simply defined customer and issuer entity in data to avoid lot of typing;
        invoice["customer"] = data["customer"][invoice["customer"]]
        invoice["issuer"] = data["issuer"][invoice["issuer"]]
        # we also needs to add total to items:
        total = 0
        for item in invoice["items"]:
            total += float(item["price"]) * int(item["quantity"])
        invoice["total"] = total
    return data["invoices"]