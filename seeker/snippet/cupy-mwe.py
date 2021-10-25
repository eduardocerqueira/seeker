#date: 2021-10-25T17:09:23Z
#url: https://api.github.com/gists/0fcae87071ef4b448d1b8c0f89628e33
#owner: https://api.github.com/users/brandondube

"""LOWFSC server -- simulation server for PACE."""
from io import BytesIO

from imageio import imwrite
from flask import Flask, send_file

try:
    import cupy as cp
    from cupyx.scipy import fft as cpfft

    cp.cuda.runtime.setDevice(1)  # use the second GPU, "private"
    # cp.cuda.Device(1).use()
    # print('Using GPU', cp.cuda.runtime.getDevice())
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
except ImportError:
    pass


app = Flask('cupy-dev1-memleak-mwe')


@app.route('/wrong-gpu-and-memory-leak', methods=['GET'])
def foobar():
    a = cp.random.rand(512,512)
    b = cpfft.fftshift(cpfft.fft2(cpfft.ifftshift(a)))
    c = b.get()
    buf = BytesIO()
    imwrite(buf, c, format='tiff')
    buf.seek(0)
    return send_file(buf, mimetype='image/tiff')


