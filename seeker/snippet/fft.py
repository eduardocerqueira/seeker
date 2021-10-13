#date: 2021-10-13T17:14:23Z
#url: https://api.github.com/gists/aac6c48806ce4f5caf585d6bd87301c7
#owner: https://api.github.com/users/benbrowning1

def FFT(arr):
    n = len(arr)

    if n == 1:
        return arr

    w =( (np.e) ** (complex(real=0, imag=-1) * 2 * np.pi / n))

    EvenArr, OddArr = arr[::2], arr[1::2]

    YE, YO = FFT(EvenArr), FFT(OddArr)

    y = np.zeros(n, dtype=complex)

    for j in range(n // 2):
        t = (w ** (j)) * YO[j]
        y[j] = YE[j] + t
        y[j + n // 2] = YE[j] - t
    return y
