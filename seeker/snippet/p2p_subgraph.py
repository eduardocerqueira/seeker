#date: 2022-06-27T17:11:41Z
#url: https://api.github.com/gists/a32ffb0359ec6556066c5df2a9860aae
#owner: https://api.github.com/users/tomasonjo

largestComponentId = (
    gds.graph.streamNodeProperty(G, "wcc")
    .groupby("propertyValue")
    .size()
    .to_frame("componentSize")
    .reset_index()
    .sort_values(by="componentSize", ascending=False)["propertyValue"][0]
)

largestComponentGraph, res = gds.beta.graph.project.subgraph(
    "largestComponent", G, f"n.wcc = {largestComponentId}", "*"
)