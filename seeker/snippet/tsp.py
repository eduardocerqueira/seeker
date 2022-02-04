#date: 2022-02-04T17:05:44Z
#url: https://api.github.com/gists/f25faf1f954fdb5653804df42cfbc1e6
#owner: https://api.github.com/users/Playdead1709

import tsp
matrix=[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
r=range(len(matrix))
shortestpath=((i,j):matrix[i][j] for i in r for j in r)
print(tsp.tsp(r,shortestpath))
