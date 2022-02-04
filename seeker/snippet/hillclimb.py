#date: 2022-02-04T17:05:44Z
#url: https://api.github.com/gists/f25faf1f954fdb5653804df42cfbc1e6
#owner: https://api.github.com/users/Playdead1709

Succlist={'A':['B',3],['C',2]],
          'B':[['D',2]['E',3]],
          'C':[['F',2],['G',4]],
          'D':[['H',1],['I',99]],
          'F':[['J',99]],
          'G':[['K',99],['L',3]]}
start='A'
closed=list()
def hill_climb(start):
  global closed
  N=start
  child=movegen(N)
  Sort(child)
  N=[start,5]
  Print('\n start=', N)
  Print('sorted childlist=',child)
  Newnode=child[0]
  CLOSED=[N]
While heu(newnode)<=heu(N):
  N=newnode
  print[“N=”,N)
  CLOSED=append(CLOSED,[N])
  child=movegen(N[0]]
  Sort(child)
  print(“sorted child list=”,child)
  print(“CLOSED=”,CLOSED)
  Newnode=child[0]
  Closed=CLOSED