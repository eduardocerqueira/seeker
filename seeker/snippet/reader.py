#date: 2023-02-07T16:44:07Z
#url: https://api.github.com/gists/8680ae2e951a6d319f5e5dedbe93c366
#owner: https://api.github.com/users/RJaBi

import numpy as np
import sys
import os
from ctypes import *

def main():
    fDir = '/home/RJaBi/Documents/2023/wloops/openqcd/data/ms10/dat'
    fFile = 'Gen2l_64x32.ms10.dat'
    fFP = os.path.join(fDir, fFile)
    """
    For each configuration, all Wilson loops of size (t x r) with t=1,...,mwlt
    and r=1,...,mwlr, and smearing levels sl1, sl2 for the space-like links at the
    two space-like edges with sl1,sl2=nns[0],...,nns[msl-1] are computed. The Wilson
    loops are summed over the three spacial directions and the spatial volume, but
    not over time (see modules/wloop/wloop.c and modules/wloop/smear.c). 
    """
    
    class headerStruct(Structure):
        """
        The header data
        """
        _fields_ = [('wls', c_int),
                    ('msl', c_int),
                    ('mwlt', c_int),
                    ('mwlr', c_int),
                    ('tmax', c_int)]

    with open(fFP, 'rb') as f:
        header = headerStruct()
        header_size = f.readinto(header)
        header = [header.wls, header.msl, header.mwlt, header.mwlr, header.tmax]

        # The structs to load the actual data are placed here
        # as we need the number of entries for each config wls = header[0]
        class conf(Structure):
            """
            This is the actual structure of the data for each conf
            """
            wl_mat = [('wl_mat', header[0] * c_double)]
            _fields_ = [('nc', c_int)] + wl_mat

        class ncStruct(Structure):
            """
            This is just the configuration number
            """
            _fields_ = [('nc', c_int)]
            
        class wl_matStruct(Structure):
            """
            The data for a single configuration. It is arranged
            off = (sl1+msl*(sl2+msl*(tt+mwlt*(rr+mwlr*x0))))
            """
            _fields_ = [('wl_mat', header[0] * c_double)]

        # store a list of each configs data
        data = []
        # a list of the configurationis used
        iconList = []
        # We have three 'structs' here
        # The whole, the config num, the data
        # THis is as I was having difficulties getting the config num
        # correct using just the whole
        allc = conf()
        allNC = ncStruct()
        allwl = wl_matStruct()
        data = []
        while f.readinto(allc) == sizeof(allc):
            # So we have no read the whole into allc
            # and advanced the file pointer
            # So get the file pointer
            currOff = f.tell()
            # Go back to start of the whole
            f.seek(-sizeof(allc), 1)
            # Read the config numb
            f.readinto(allNC)
            iconList.append(allNC.nc)
            # Read the data
            f.readinto(allwl)
            data.append(allwl.wl_mat)