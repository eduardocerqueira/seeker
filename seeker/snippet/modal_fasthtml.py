#date: 2024-08-15T16:54:47Z
#url: https://api.github.com/gists/e28466e5091792336a0cad5205e0644b
#owner: https://api.github.com/users/hgbrian

"""
pip install modal python-fasthtml
modal serve modal_fasthtml.py
"""

import json
import modal
from modal import asgi_app
from fasthtml.common import fast_app, Script, Titled, Div

fast_html_app, rt = fast_app(hdrs=(Script(src="https://cdn.plot.ly/plotly-2.32.0.min.js"),))

data = json.dumps({
    "data": [{"x": [1, 2, 3, 4],"type": "scatter"},
             {"x": [1, 2, 3, 4],"y": [16, 5, 11, 9],"type": "scatter"}],
    "title": "Plotly chart in FastHTML ",
    "description": "This is a demo dashboard",
    "type": "scatter"
})

@rt("/")
def get():
    return Titled("Chart Demo", Div(id="myDiv"),
        Script(f"var data = {data}; Plotly.newPlot('myDiv', data);"))

app = modal.App()

@app.function(
    image=modal.Image.debian_slim().pip_install("python-fasthtml"),
    gpu=False,
)
@asgi_app()
def fasthtml_asgi():
    return fast_html_app

if __name__ == "__main__":
    modal.serve(fast_html_app)