#date: 2024-04-18T17:06:37Z
#url: https://api.github.com/gists/7ad530644a9e9fd09f9d99c6d6fa3c20
#owner: https://api.github.com/users/rodff

import os
import sys
import configparser

def main():
    # Configuration file
    conf_fn = str(sys.argv[1])

    # Read configuration
    cfg = configparser.ConfigParser()
    cfg.read(conf_fn)
    Dic = cfg['DEFAULT']
    
    outpath = str(Dic['save_directory'])
    N = int(Dic['num_particles'])
    m = float(Dic['constant'])

    print('Using configurations from {} \n'.format(conf_fn))
    

    print('Pre-processing... \n')
    os.system('python preprocess_1.py {}'.format(N))
    for x in range(N):
        for y in range(N):
            os.system('python preprocess_2.py {} {}'.format(x,y))
    

    print('Analyzing... \n')
    os.system('python simple_analysis.py')
    for i in [8,16,32]:
        os.system('python full_analysis.py {} {}'.format(m,i))


    print('Plotting... \n')
    os.system('python plot_diagrams.py {}'.format(outpath)))
    os.system('python plot_stat_tests.py {}'.format(outpath)))
        
    
if __name__ == '__main__':
    main()