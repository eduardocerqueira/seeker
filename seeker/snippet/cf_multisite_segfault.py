#date: 2023-07-24T16:40:38Z
#url: https://api.github.com/gists/7d4480ce9f5926de4b2e44c42caaa868
#owner: https://api.github.com/users/mducle

# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
import sys, os

from CrystalField import CrystalField, PointCharge, ResolutionModel, CrystalFieldFit, Background, Function
from pychop.Instruments import Instrument

tb1 = {'B20': -0.2431330412981789, 'B22': -0.17569696468779034, 'IB22': 0.19648505887037993, 'B40': -0.000886093463241086, 'B42': -0.016200141304394847, 'IB42': 0.017849375647540782, 'B44': -0.0011331370797592647, 'IB44': -0.014528896930898594, 'B60': 5.487217052410268e-06, 'B62': 3.0859080066753127e-06, 'IB62': -3.1901616555494786e-06, 'B64': -2.3561324645561506e-06, 'IB64': 1.8508497796793596e-05, 'B66': 2.2094823318064904e-05, 'IB66': 1.273979589242308e-05}
tb2 = {'B20': -0.005635326736244166, 'B22': 0.9342119433862551, 'IB22': 0.16154603310566612, 'B40': -0.0009350468066064394, 'B42': 0.01619027473906539, 'IB42': -0.014866637051791342, 'B44': 0.006361657737804525, 'IB44': -0.010630085692637816, 'B60': 5.011125389218243e-06, 'B62': 9.619136669456379e-06, 'IB62': 5.574095093138739e-06, 'B64': 1.0647772671681482e-05, 'IB64': 2.7759271573561253e-05, 'B66': 1.6305270683919554e-05, 'IB66': -1.2197676918277416e-05}

merlin = Instrument('MERLIN', 'G', 300.)
merlin.setEi(82.)
resmod1 = ResolutionModel(merlin.getResolution, xstart=-10, xend=81.9, accuracy=0.01)

print('Site 1')
cf1 = CrystalField('Tb', 'C2', Temperature=7, FWHM=0.1, **tb1)
cf1.PeakShape = 'Gaussian'
cf1.ResolutionModel = resmod1
xx, yy = cf1.getSpectrum(x_range=(0, 100))
cfref1 = CreateWorkspace(xx, yy, np.sqrt(yy)+np.sqrt(10))

print(cf1.getPeakList())
print(cf1.getEigenvalues())

print()
print('Site 2')
cf2 = CrystalField('Tb', 'C2', Temperature=7, FWHM=0.1, **tb2)
cf2.PeakShape = 'Gaussian'
cf2.ResolutionModel = resmod1
xx, yy = cf2.getSpectrum(x_range=(0, 100))
cfref2 = CreateWorkspace(xx, yy, np.sqrt(yy)+np.sqrt(10))

print(cf2.getPeakList())
print(cf2.getEigenvalues())

xx = cfref1.extractX()
randnoise = CreateWorkspace(xx, np.random.rand(xx.shape[1]) * 10)
cfref = cfref1 + cfref2 + 10 + randnoise

cf = cf1 + cf2

fit = CrystalFieldFit(Model=cf, InputWorkspace=cfref, MaxIterations=0, Output='fit')
fit.fit()