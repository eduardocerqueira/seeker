#date: 2025-05-21T16:41:22Z
#url: https://api.github.com/gists/b401c5c5676e48c85ab069e9ad46313e
#owner: https://api.github.com/users/CyanBread

import json
import markdown


def find_path_for_function(functionName: str, line: str) -> str:
    """attempt to find a function call in the line and return its first string parameter"""
    query = "." + functionName + "("
    query_index = line.find(query)
    if query_index != -1:
        line_clean = line.replace("'", '"')
        path_start = query_index + len(query) + 1
        path_end = line_clean.find('"', path_start)
        return line_clean[path_start:path_end]
    else:
        return None


def create_CSV_XML(path: str):
    return f"""<p>
<ac:structured-macro ac:name="csv-table" ac:schema-version="1">
<ac:parameter ac:name="attachment">{path}</ac:parameter>
<ac:parameter ac:name="source">attachment</ac:parameter>
</ac:structured-macro>
</p>"""


def create_SVG_XML(path: str, page_title: str, wiki_title="TUMddmlab"):
    return f"""<p>
<ac:image>
<ri:attachment ri:filename="{path}">
<ri:page ri:space-key="{wiki_title}" ri:content-title="{page_title}" />
</ri:attachment>
</ac:image>
</p>"""


def convert_jupyter_to_confluence(jupyter_path: str, output_path: str, page_title: str):
    """
    converts a jupyter notebook to pseudo-html confluence format.

    **IMPORTANT**: The page title of the confluence page is needed to correctly locate images.

    All markdown cells are parsed.
    All text printed using `display(Markdown(""))` is parsed.
    References are created for all SVG plots that are saved using `plt.savefig("")`.
    References are created for all CSV tables that are saved using `df.to_csv("")`.
    Plots and tables need to be uploaded as attachments to the confluence page.(bulk-upload is possible)
    """
    with open(jupyter_path, "r") as json_data:
        with open(output_path, "w") as out:
            nb = json.load(json_data)
            for cell in nb["cells"]:
                # print entire markdown cells
                if cell["cell_type"] == "markdown":
                    for line in cell["source"]:
                        out.write(markdown.markdown(line) + "\n")
                # print select parts of code cells
                if cell["cell_type"] == "code":
                    # print markdown output (use display(Markdown("I want this in my output")))
                    for output in cell["outputs"]:
                        if output["output_type"] == "display_data":
                            if "text/markdown" in output["data"]:
                                for line in output["data"]["text/markdown"]:
                                    out.write(markdown.markdown(line) + "\n")
                    # Add graphs and tables that were saved in cell blocks.
                    line: str
                    for line in cell["source"]:
                        plot_path = find_path_for_function("savefig", line)
                        if plot_path:
                            out.write(create_SVG_XML(plot_path, page_title) + "\n")
                        csv_path = find_path_for_function("to_csv", line)
                        if csv_path:
                            out.write(create_CSV_XML(csv_path) + "\n")


convert_jupyter_to_confluence(
    "PurchaseExploration/PurchasesExploration.ipynb", "out.xml", "Jupyter Export Test"
)
