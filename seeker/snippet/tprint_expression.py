#date: 2023-03-06T16:41:05Z
#url: https://api.github.com/gists/05b605d9e770622c80567e013d17fba4
#owner: https://api.github.com/users/UmeSiyah

"""
Print in the terminal all knob whom have a expression
and print the expression
"""
nuke.tprint(f"\n{'~'*80}")
for node in nuke.allNodes():
    for knob_name in node.knobs():
        knob = node.knob(knob_name)
        if not knob.hasExpression():
            continue
        nuke.tprint(knob)
        nuke.tprint(knob.toScript())
        nuke.tprint('~')
