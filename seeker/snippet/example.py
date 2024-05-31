#date: 2024-05-31T17:08:08Z
#url: https://api.github.com/gists/146fbdc9b471f22d432488df54ae9554
#owner: https://api.github.com/users/kazuki0824


from flowtest_graphtools.tcloader.utils import Transition, generate_expr
from flowtest_graphtools.pathplan.planner2 import LayeredGraph
edges = {
    'i': Transition(fro='ア', to='イ', condition=generate_expr("!Flow_003&Flow_001"), tc_on=True, manual=False),
    'ii': Transition(fro='イ', to='ウ', condition=generate_expr("Flow_002"), tc_on=True, manual=False),
    'iii': Transition(fro='ア', to='ウ', condition=generate_expr("Flow_001"), tc_on=True, manual=False),
    'iv': Transition(fro='イ', to='ウ', condition=generate_expr("!Flow_003"), tc_on=True, manual=False),
    'v': Transition(fro='イ', to='ウ', condition=generate_expr("!Flow_003&Flow_001"), tc_on=True, manual=False)
}

lg = LayeredGraph(edges, "ア", enabled_flow_pats={"A": generate_expr("Flow_001")})