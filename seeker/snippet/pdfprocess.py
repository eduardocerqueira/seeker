#date: 2021-09-16T16:54:12Z
#url: https://api.github.com/gists/3a5326e502f9bfdd4cf1b502bf9f5f76
#owner: https://api.github.com/users/opalczynski

import weasyprint
import yaml
from jinja2 import Template


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


def get_template(template_path="./template.html"):
    with open(template_path, "r") as f:
        return Template(f.read())


def render_pdf(invoices):
    template = get_template()
    for invoice in invoices:
        rendered = template.render(**invoice)
        html = weasyprint.HTML(string=rendered, base_url="/")
        css = weasyprint.CSS(filename="./template.css")
        with open(f"output/{invoice['number']}-{invoice['customer']['name']}.pdf", "wb") as f:
            html.write_pdf(f, stylesheets=[css])


invoices = get_data()
render_pdf(invoices=invoices)
