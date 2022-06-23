#date: 2022-06-23T16:58:44Z
#url: https://api.github.com/gists/c801565e499be605cea1283a5984b4c3
#owner: https://api.github.com/users/mateon1

def varchar(lit, seen=None):
    if seen:
        return {(True, True): "X", (True, False): "#", (False, True): ".", (False, False): "?"}[(lit in seen, -lit in seen)]
    else:
        v = slv.val(lit)
        if v is True: return "#"
        if v is False: return "."
        assert v is None
        if slv.resolvemerge(lit) != lit: return "/"
        if len(slv.lit_refs[lit]) + len(slv.lit_refs[-lit]) == 0 and len(slv.idag.b[slv.idag.base(lit)]) == 0 and len(slv.idag.b[slv.idag.base(-lit)]) == 0:
            return "*"
        return "?"
def format_grid(grid, val):
    for r in grid:
        out = ["%05d: "%r[0]]
        for u in r:
            out.append(varchar(u, val))
        print("".join(out))
class DummyCell:
    def __init__(self, slv):
        self.solver = slv
    def pretty(self, seen=None):
        if seen is None:
            seen = {v if self.solver.val(v) else -v for v in range(1,len(self.solver.lit_refs)) if self.solver.val(v) is not None}
        print("== RULES ==")
        format_grid([[r0[0], r0[1], *r0[2], r1[0], r1[1], *r1[2]] for r0,r1 in rules], seen)
        print("== EVOLUTION ==")
        format_grid([t[1] + t[0] + t[2] + [w,d] + s + [h] for t, (w,d,s,h) in zip(tapes, trans)] + [tapes[-1][1] + tapes[-1][0] + tapes[-1][2]], seen)


slv = Solver()
cell = DummyCell(slv)

# --- #

STATES = 3
N = 99
G = N//2+1+12

rules = [((slv.allocate_var(), slv.allocate_var(), slv.allocate_vars(STATES+1)), (slv.allocate_var(), slv.allocate_var(), slv.allocate_vars(STATES+1))) for _ in range(STATES)]

slv.unit(rules[0][0][0]) # first rule always writes 1 (otherwise can be reordered, or no rule writes 1)
slv.unit(rules[0][0][1]) # first rule always goes right (symmetry breaking)
slv.unit(-rules[0][0][2][0]) # first rule doesn't go to itself (trival infinite loop)

for g, (w, d, s) in enumerate(r for (r0, r1) in rules for r in (r0, r1)):
    for i in range(STATES+1):
        if i != STATES and i > g + 1: # state 0: left can refer to 1 or self, right can refer up to 2, state 1: left can refer to 3 or less... (binary tree / heap)
            slv.unit(-s[i])
            continue
        for j in range(i+1, STATES+1):
            slv.bclause(-s[i], -s[j])
    slv.bclause(-s[STATES], w)
    slv.bclause(-s[STATES], d) # HALT rule is always (1, R, H)
    # canonicalization: transition to state 'n' only possible if state 'n-1' reachable somewhere earlier
    for i in range(1,STATES-1):
        slv.make_clause([-s[i+1]] + [r[2][i] for (r0, r1) in rules for r in (r0, r1)][:g])

def mostone(lits):
    for i in range(len(lits)):
        for j in range(i+1, len(lits)):
            slv.bclause(-lits[i], -lits[j])

for s in range(1, STATES):
    # all states except the starting state reachable from another state
    slv.make_clause([r[2][s] for (r0, r1) in rules[:s] + rules[s+1:] for r in (r0, r1)])
    # no state always leads to itself
    slv.bclause(-rules[s][0][2][s], -rules[s][1][2][s])

slv.make_clause([-r[0][1] for r in rules[1:STATES]] + [r[0][2][STATES] for r in rules]) # one of the 'read 0' rules moves left or halts

# exactly one halting state
mostone([r[2][STATES] for (r0, r1) in rules for r in (r0, r1)])
slv.make_clause([r[2][STATES] for (r0, r1) in rules for r in (r0, r1)])

def tape_entry(N=N):
    t = slv.allocate_vars(N)
    s = slv.allocate_vars(STATES+1)
    h = slv.allocate_vars(N)
    mostone(s)
    mostone(h)
    slv.make_clause(s)
    slv.make_clause(h)
    return t, s, h

tapes = [tape_entry()] # (tape, s, head)
trans = [] # (w, d, s, h)

for u in tapes[0][0]: slv.unit(-u)
slv.unit(tapes[0][2][N//2-1])
slv.unit(tapes[0][1][0])
slv.propagate()

def step_tapes(t1, t2, N=N):
    lt, ls, lh = t1
    ct, cs, ch = t2
    w, d = slv.allocate_var(), slv.allocate_var()
    # head value
    h = slv.allocate_var()
    for i in range(N):
        slv.make_clause([-lh[i], -lt[i], h])
        slv.make_clause([-lh[i], lt[i], -h])
    # rule
    slv.bclause(-ls[STATES], cs[STATES]) # HALT, w and d unbound
    for i in range(STATES):
        slv.make_clause([-ls[i], h, -rules[i][0][0], w])
        slv.make_clause([-ls[i], h, rules[i][0][0], -w])
        slv.make_clause([-ls[i], h, -rules[i][0][1], d])
        slv.make_clause([-ls[i], h, rules[i][0][1], -d])
        for j in range(STATES+1):
            slv.make_clause([-ls[i], h, -rules[i][0][2][j], cs[j]])
            slv.make_clause([-ls[i], h, rules[i][0][2][j], -cs[j]])
        slv.make_clause([-ls[i], -h, -rules[i][1][0], w])
        slv.make_clause([-ls[i], -h, rules[i][1][0], -w])
        slv.make_clause([-ls[i], -h, -rules[i][1][1], d])
        slv.make_clause([-ls[i], -h, rules[i][1][1], -d])
        for j in range(STATES+1):
            slv.make_clause([-ls[i], -h, -rules[i][1][2][j], cs[j]])
            slv.make_clause([-ls[i], -h, rules[i][1][2][j], -cs[j]])
    # head movement
    for i in range(1, N-1):
        slv.make_clause([ls[STATES], -ch[i], lh[i-1], lh[i+1]])
        slv.make_clause([ls[STATES], -lh[i], ch[i-1], ch[i+1]])
        slv.make_clause([ls[STATES], -lh[i], -d, ch[i+1]])
        slv.make_clause([ls[STATES], -lh[i], d, ch[i-1]])
    slv.make_clause([ls[STATES], -ch[0], lh[1]])
    slv.make_clause([ls[STATES], -lh[0], ch[1]]) # border conditions
    slv.make_clause([ls[STATES], -lh[0], d])
    slv.make_clause([ls[STATES], -ch[-1], lh[-2]])
    slv.make_clause([ls[STATES], -lh[-1], ch[-2]])
    slv.make_clause([ls[STATES], -lh[-1], -d])
    # tape write
    for i in range(N):
        slv.make_clause([lh[i], -lt[i], ct[i]]) # head not on tape position -> unchanged
        slv.make_clause([lh[i], lt[i], -ct[i]])
        slv.make_clause([-lh[i], -w, ct[i]]) # head on tape position -> gets written value
        slv.make_clause([-lh[i], w, -ct[i]])
        slv.make_clause([-ls[STATES], -lt[i], ct[i]]) # HALT state
        slv.make_clause([-ls[STATES], lt[i], -ct[i]])
        slv.make_clause([-ls[STATES], -lh[i], ch[i]])
        slv.make_clause([-ls[STATES], lh[i], -ch[i]])
    return (w, d, cs, h)

def append_tape(N=N):
    t1 = tapes[-1]
    t2 = tape_entry(N)
    tapes.append(t2)
    trans.append(step_tapes(t1, t2, N))

#for i in range(G): append_tape()

# halt

#HN = 15
#HG = HN//2+1
#tapes.append(tape_entry(HN))
#trans.append((-1, -1, [], -1))
#
#for i in range(HG): append_tape(HN)
#
#slv.unit(tapes[-1][1][STATES])
#slv.unit(-tapes[-2][1][STATES])
#slv.unit(tapes[-1][2][HN//2+1])
