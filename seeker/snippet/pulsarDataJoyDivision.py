#date: 2023-07-07T16:59:59Z
#url: https://api.github.com/gists/991b4725b0ce01317b088054560b2e45
#owner: https://api.github.com/users/erikson1970

import numpy as np
import matplotlib.pyplot as plt
import getopt,sys
from json import loads,loads

def plot_pulsar_data(filename=None,nSamps:int=1000,nLines:int=100, seed:int=10000,
                     seriesTerms:list[tuple[int]]=[(-1.5,1.6),(2,0.5),(3,1.2)],
                     syncBumpRandomTerm:float=0.3,
                     noiseTerms:list[float]=[15.0,0.004,2.0,1.2],
                     smoothingTerms:dict={"smaPeriod":3,"outTerms":0}):
    """Make a random pulsar plot a-la Joy Division Album cover"""
    np.random.seed(seed=seed)
    fig, ax = plt.subplots(figsize=(15,15))
    x = np.linspace(-14, 14, nSamps)
    syncBump=lambda xx,aa,bb,cc: (bb * np.sin(cc*(xx - aa))/(xx-aa))
    smoothNoise=lambda data, smaPeriod, outTerms:  [None if outTerms is None else outTerms*np.mean(data[: smaPeriod])]* (smaPeriod - 1)  + [np.mean(data[i - smaPeriod + 1: i + 1]) for i in range(len(data))[smaPeriod - 1:]]
    for w in np.linspace(30.4, 0, nLines ):
        adjAll=4*np.random.rand(6)
        allY=np.zeros(len(x))
        for scY,scYamp in seriesTerms:
          allY+=syncBump(x,scY+syncBumpRandomTerm*np.random.randn(),scYamp,0.75 + np.random.rand())
        y_lpf = (np.convolve(allY, smoothNoise(np.random.randn(50)/noiseTerms[0]+noiseTerms[1],**smoothingTerms) , mode='same') ** noiseTerms[2])/noiseTerms[3]
        y_lpf += smoothNoise(np.random.randn(len(x))* 0.2,20,1.0)
        ax.fill_between(x, w+y_lpf, 0, facecolor='black',edgecolor='white')
    ax.set_axis_off()
    if filename:
        ext=filename.split(".")[-1]
        if ext in "eps,jpeg,jpg,pdf,pgf,png,ps,raw,rgba,svg,svgz,tif,tiff,webp":
          plt.savefig(filename)
        else:
          print(f"Unsupported image filename extension '{ext}' - can't save image")
    else:
        plt.show()

if __name__=="__main__":
  argumentList = sys.argv[1:]
  #  argumentList = ['-t','{"syncBumpRandomTerm": 2.0, "noiseTerms": [15.0, 0.004, 2.0, 1.2]}']
  plotDataParms={"filename":None,"seed":np.random.randint(100000000)}
  # Options
  options,long_options = ("hf:t:s:",["help", "filename=", "terms=", "seed="])
  try:
      # Parsing argument
      arguments, values = getopt.getopt(argumentList, options, long_options)
      
      # checking each argument
      for currentArgument, currentValue in arguments:
  
          if currentArgument in ("-h", "--help"):
              print ("USAGE: -s set params, -f set output file(optional)")
              
          elif currentArgument in ("-f", "--filename="):
              plotDataParms["filename"]=currentValue
              
          elif currentArgument in ("-s", "--seed="):
              plotDataParms["seed"]=int(currentValue)
              
          elif currentArgument in ("-t", "--terms="):
            for k,v in loads(currentValue).items():
              plotDataParms[k] = v
      print(f"plotting with following parameters: {plotDataParms}")
      plot_pulsar_data(**plotDataParms)
          
  except getopt.error as err:
      print (str(err))