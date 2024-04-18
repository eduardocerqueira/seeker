#date: 2024-04-18T17:06:37Z
#url: https://api.github.com/gists/7ad530644a9e9fd09f9d99c6d6fa3c20
#owner: https://api.github.com/users/rodff

import os
import sys
import configparser

def main():
    
    outpath = '/home/fulano/project/'
    N = 1000
    m = 12.34567

    print('Pre-processing... \n')
    for x in range(N):
        os.system('python preprocess.py {}'.format(x))
    

    print('Analyzing... \n')
    for i in [8,16,32]:
        os.system('python analysis.py {} {}'.format(m,i))


    print('Plotting... \n')
    os.system('python plot.py {}'.format(outpath))
        
if __name__ == '__main__':
    main()