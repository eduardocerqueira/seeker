#date: 2021-11-23T17:13:29Z
#url: https://api.github.com/gists/8764e1fb4230a1efe57b2443783e1f5a
#owner: https://api.github.com/users/kn1ghtf1re

import os
import numpy as np
import scipy.io.wavfile
import scipy.signal

if __name__ == '__main__':

    dir = 'test/audio/'
    newdir = 'testnew/audio/'
    files = os.listdir(dir)
    samplerate = 22050

    counter = 0
    for f in files:
        counter += 1
        rate, data = scipy.io.wavfile.read(dir+f)
        print('Count: {}, File: {} | Rate: {} | Samples: {}'.format(counter, f, rate, len(data)))
        
        # Calculate new sample
        samples = len(data) * int(22050/8000)
        _ = scipy.signal.resample(data, samples).astype(np.int16)
        n = len(_)
        assert n == samples
        # value = _[-1]
        # __ = np.append(_, value)
        # n = len(__)
        # print(n)
        # assert n == samplerate

        scipy.io.wavfile.write(newdir+f, samplerate, _)
