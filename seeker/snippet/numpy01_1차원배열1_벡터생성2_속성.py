#date: 2021-11-17T16:51:11Z
#url: https://api.github.com/gists/3452f645eb100eb60bd9b8033d18f1b7
#owner: https://api.github.com/users/chinshin5513

import numpy as np # 별칭 주기, 앞으로는 np. 으로 사용이 가능하다.

# 1. 1차원 배열인 벡터 생성
list_value = [10, 20, 30, 40]  # 파이썬 타입, 타입명: list
v1 = np.array(list_value)  # 이터러블(리스트, 튜플 등)
print(v1)  # 넘파이 타입(쉼표 없음), 이게 바로 벡터 타입명: ndarray

'''
print(dir(v1))  # 넘파이
'all', 'any', 'argmax', 'argmin', 'argpartition', 'argsort', 'astype', 'base', 'byteswap', 'choose', 'clip', 
'compress', 'conj', 'conjugate', 'copy', 'ctypes', 'cumprod', 'cumsum', 'data', 'diagonal', 'dot', 'dtype', 'dump', 
'dumps', 'fill', 'flags', 'flat', 'flatten', 'getfield', 'imag', 'item', 'itemset', 'itemsize', 'max', 'mean', 'min',
'nbytes', 'ndim', 'newbyteorder', 'nonzero', 'partition', 'prod', 'ptp', 'put', 'ravel', 'real', 'repeat', 'reshape', 
'resize', 'round', 'searchsorted', 'setfield', 'setflags', 'shape', 'size', 'sort', 'squeeze', 'std', 'strides', 
'sum', 'swapaxes', 'take', 'tobytes', 'tofile', 'tolist', 'tostring', 'trace', 'transpose', 'var', 'view'
'''
