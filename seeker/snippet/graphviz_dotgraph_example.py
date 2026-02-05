#date: 2026-02-05T17:28:27Z
#url: https://api.github.com/gists/900bc09d4a7424c1e3dec7e5c03a4a06
#owner: https://api.github.com/users/neerajadsul

import graphviz
from IPython import display
from PIL import Image

syst_search = r"Systematic Hiking Location Search\lDatabase Review\l"
dup_removal = "Duplicate Hiking Location Removal\nCleansing Redundant Spots"
dedup = "Deduplication of Hiking Locations\nRemoving Duplicate Trails"
s1 = "Stage 1: Location Screening\nInitial Trail Assessment\n"
excluded = r"Excluded Hiking Locations\lFiltered Out Trails\l"
included = "Included Hiking Locations\nSelected Trails"
s2 = "Stage 2: Detailed Location Evaluation\nTrail Assessment Phase"
excluded_s2 = r"Excluded from Final Selection\lTrails Removed at Stage 2\l"
included_s2 = r"Final Selected Hiking Locations\lApproved Trails\l"
manual_search = "Manual Hiking Location Search\nSupplementary Trail Identification"

# e = graphviz.Graph(engine="neato", filename="test")
e = graphviz.Digraph(filename="test")#, graph_attr={"mode": "sgd"})
e.node(name="SS", label=syst_search, shape="box")
e.node(name="DUP", label=dup_removal, shape="diamond", style="filled", color="lightgrey")
e.edge("SS", "DUP")
e.node("DEDUP", label=dedup, shape="box")
e.edge("DUP", "DEDUP")
e.node("S1", label=f"{s1}", shape="diamond", style="filled", color="lightgrey")
e.node("EXCLUDED", label=excluded, shape="box")
e.edge("S1", "EXCLUDED", label="exclude")
e.node("INCLUDED", label=included, shape="box")
e.node("S1", label=s1, shape="diamond")
e.node("EXCLUDED", label=excluded, shape="box")
e.node("INCLUDED", label=included, shape="box")
for arrow in [("DEDUP", "S1"), ("S1", "INCLUDED")]:
    e.edge(arrow[0], arrow[1])
e.node("S2", label=s2, shape="diamond", style="filled", color="lightgrey")
e.node("EXCL_2", label=excluded_s2, shape="box")
e.node("MANUAL_SEARCH", label=manual_search, shape="box")
e.node("INCL_2", label=included_s2, shape="box")
for arrow in [("INCLUDED", "S2"), ("S2", "EXCL_2"), ("S2", "INCL_2"), ("MANUAL_SEARCH","S2")]:
    e.edge(arrow[0], arrow[1])

e.render(format="png")
Image.open("test.png")