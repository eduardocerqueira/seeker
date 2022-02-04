#date: 2022-02-04T17:05:44Z
#url: https://api.github.com/gists/f25faf1f954fdb5653804df42cfbc1e6
#owner: https://api.github.com/users/Playdead1709

def TowerOfHanoi(n , from_rod, to_rod, aux_rod):
    if n == 0:
        return
    TowerOfHanoi(n-1, from_rod, aux_rod, to_rod)
    print("Move disk",n,"from rod",from_rod,"to rod",to_rod)
    TowerOfHanoi(n-1, aux_rod, to_rod, from_rod)
         
n = 4
TowerOfHanoi(n, 'A', 'C', 'B')