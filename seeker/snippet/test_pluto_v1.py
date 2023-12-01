#date: 2023-12-01T17:03:30Z
#url: https://api.github.com/gists/f4a4ca3c265c2ba1d3de8c2b8251847e
#owner: https://api.github.com/users/tfcollins

import iio
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import time

def gen_data(fs, fc):
    N = 1024
    ts = 1 / float(fs)
    t = np.arange(0, N * ts, ts)
    i = np.cos(2 * np.pi * t * fc) * 2 ** 14
    q = np.sin(2 * np.pi * t * fc) * 2 ** 14
    iq = i + 1j * q
    return iq

def test_complex_buffer():
    fs = 4e6
    lo = 1e9
    fcd = 1e5

    ctx = iio.Context('ip:analog.local')
    dev = ctx.find_device('ad9361-phy')
    dev_rx = ctx.find_device('cf-ad9361-lpc')
    dev_tx = ctx.find_device('cf-ad9361-dds-core-lpc')

    chan = dev.find_channel('voltage0')
    chan.attrs['sampling_frequency'].value = str(int(fs))
    chan.attrs['rf_bandwidth'].value = str(int(fs))

    achan = dev.find_channel('altvoltage0', True)
    achan.attrs['frequency'].value = str(int(lo))
    achan = dev.find_channel('altvoltage1', True)
    achan.attrs['frequency'].value = str(int(lo))

    dev.debug_attrs['loopback'].value = '1'

    ## DDS
    for N in [1]:
        for IQ in ['I', 'Q']:
            chan = f'TX1_{IQ}_F{N}'
            dds = dev_tx.find_channel(chan, True)
            if not dds:
                raise Exception(f"Could not find channel {chan}")
            dds.attrs['frequency'].value = str(int(fcd))
            dds.attrs['scale'].value = '1.0'
            if IQ == 'I':
                dds.attrs['phase'].value = '90000'
            else:
                dds.attrs['phase'].value = '0.0'


    ## Buffer stuff

    # RX Side
    chan1 = dev_rx.find_channel('voltage0')
    chan2 = dev_rx.find_channel('voltage1')
    mask = iio.ChannelsMask(dev_rx)
    mask.channels = [chan1, chan2]

    buf = iio.Buffer(dev_rx, mask)
    stream = iio.Stream(buf, 1024)

    # TX Side
    chan1_tx = dev_tx.find_channel('voltage0', True)
    chan2_tx = dev_tx.find_channel('voltage1', True)
    mask_tx = iio.ChannelsMask(dev_tx)
    mask_tx.channels = [chan1_tx, chan2_tx]

    # Create a sinewave waveform
    fc = 1e5
    iq = gen_data(fs, fc)

    # Send data
    buf_tx = iio.Buffer(dev_tx, mask_tx)

    ## Stream version (Does not support cyclic?)
    # tx_stream = iio.Stream(buf_tx, 1024)
    # block_tx = next(tx_stream)
    # block_tx.write(iq)
    # block_tx.enqueue(None, True)

    # Block version
    block_tx = iio.Block(buf_tx, len(iq))
    # convert iq to interleaved int16 byte array
    iqb = np.array(iq, dtype=np.int16)
    iqb = np.stack((iqb.real, iqb.imag), axis=-1)
    iqb = iqb.flatten()
    iqb = bytearray(iqb)
    block_tx.write(iqb)
    block_tx.enqueue(None, True)

    for r in range(20):
        block = next(stream)

        # Single buffer read
        if False:
            x = np.frombuffer(block.read(), dtype=np.int16)
            x = x[0::2] + 1j*x[1::2]
        else:

            # Buffer read by channel
            re = mask.channels[0].read(block)
            re = np.frombuffer(re, dtype=np.int16)
            im = mask.channels[1].read(block)
            im = np.frombuffer(im, dtype=np.int16)
            x = re + 1j*im

        f, Pxx_den = signal.periodogram(x, fs)
        plt.clf()
        plt.semilogy(f, Pxx_den)
        plt.ylim([1e-7, 1e2])
        plt.xlabel("frequency [Hz]")
        plt.ylabel("PSD [V**2/Hz]")
        plt.draw()
        plt.pause(0.05)
        time.sleep(0.1)

    plt.show()
    

if __name__ == '__main__':
    test_complex_buffer()