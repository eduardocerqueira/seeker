#date: 2021-12-21T16:55:30Z
#url: https://api.github.com/gists/fa60d5f5f7e9ba8df9a8e0a8900c2918
#owner: https://api.github.com/users/gesellkammer

#!/usr/bin/env python
# coding: utf-8

# ## Resample All Frames to Ensure Same Number of Matrix Rows

# #### v.01 — Louis Goldford (2021) — This version does not write matrix data properly.

# In[40]:


from pysdif import *
# import pandas as pd
import numpy as np
import os


# In[41]:


######### USER DEFINED VARIABLES

infileName = "Chelsea-debut_norm-hrm-v01-q150.sdif"
targetRowCount = 150 # sets target row count from pm2 analysis (i.e. ideal number of partials)

#########

infile = SdifFile(infileName)
inFileName, inFileExtension = infileName.rsplit(".", 1)
outFileName = str(inFileName) + ".resampled.v01." + str(inFileExtension)
textOutFileName = str(inFileName) + ".resampled.v01.txt"
print("made outfile:", outFileName)

outfile = SdifFile(outFileName, "w").clone_definitions(infile)
currentPath = os.getcwd()
outFileCompletePath = os.path.join(currentPath, outFileName)
# textOutFileCompletePath = os.path.join(currentPath, textOutFileName)

targetRowCountIdx = targetRowCount - 1
targetDomain = np.linspace(0, targetRowCountIdx, targetRowCount)
# print(len(targetDomain), "linspace is:", targetDomain)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


# In[42]:


count = 0

for inframe in infile:
    if inframe.signature != b'1HRM':
        continue
    with outfile.new_frame(inframe.signature, inframe.time) as outframe:
        for matrix in inframe:
            sig = matrix.signature.decode('utf-8')
            rowCount = matrix.rows
            data = matrix.get_data(copy=True)
            if data.size == 0: pass # skip empty arrays if they exist; will need to include them in outfile later (none in this data since --parSkip0 was ommitted from pm2 analysis)...
            else:
                if rowCount != targetRowCount:
                    inputDomain = np.linspace(1, targetRowCountIdx, rowCount)
                    idx = data[:,0]
                    oldFreq = data[:,1]
                    oldAmp = data[:,2]
                    oldPhase = data[:,3]
                    newFreq = []
                    newIdx = targetDomain + 1
                    intermediateFreq = np.interp(newIdx, inputDomain, oldFreq)
                    for f in intermediateFreq:
                        newFreq.append(find_nearest(oldFreq, f))
#                     newFreq = np.array(newFreq) # convert to numpy array for easy comparison...
                    newAmp = np.interp(targetDomain, inputDomain, oldAmp)
                    newPhase = np.interp(targetDomain, inputDomain, oldPhase)
                    nonTranspoMatrix = np.stack((newIdx, newFreq, newAmp, newPhase), axis=0) # <==== all ndarrays
                    transpoMatrix = np.swapaxes(nonTranspoMatrix, 0, 1) # <==== shouldn't this do the transposing?? looks right when I print...
#                     print(transpoMatrix) # < === output data looks good here; using np.float32() below for safety but still returns flat/nontransposed 1D arrays.
                    m = np.float32(transpoMatrix)
                    print(m, m.shape)
                    outframe.add_matrix(sig, m) # <==== still, writes it as a flat 1D array with no inner brackets, despite same data type (ndarray), etc.
#                     same result if I use .new_frame_one_matrix()
#                     outfile.new_frame_one_matrix(frame_sig=inframe.signature, time=inframe.time, matrix_sig=matrix.signature, data=transpoMatrix)
                    count += 1
                else:
                    outframe.add_matrix(sig, data) # unmodified if matrix.rows == target value
#                     same result if I use .new_frame_one_matrix()
#                     outfile.new_frame_one_matrix(frame_sig=inframe.signature, time=inframe.time, matrix_sig=matrix.signature, data=data)
                    count += 1
#     if count > 5:
#         break
outfile.close()
print("Process completed.")
os.system("say 'Frames resampled, file cooked. Convert SDIF to text to see result.'")


# In[ ]:
