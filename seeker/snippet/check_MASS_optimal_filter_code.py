#date: 2025-04-22T17:01:35Z
#url: https://api.github.com/gists/abcf5ef30f0d8c8c93933e3ce0f5134a
#owner: https://api.github.com/users/joefowler

import numpy as np
import mass

def verify_close(x, y, rtol=1e-5, topic=None):
    if topic is not None:
        print(f"Checking {topic:20s}: ", end="")
    isclose = np.isclose(x, y, rtol=rtol)
    print(f"x={x:.4e}, y={y:.4e} are close to each other? {isclose}")
    assert isclose
    

def test_mass_5lag_filters(Maxsignal=100.0, sigma_noise=1.0, n=500):
    tau = [.05, .25]
    t = np.linspace(-1, 1, n+4)
    npre = (t < 0).sum()
    signal = (np.exp(-t/tau[1]) - np.exp(-t/tau[0]) )
    signal[t <= 0] = 0
    signal *= Maxsignal / signal.max()
    truncated_signal = signal[2:-2]

    noise_covar = np.zeros(n)
    noise_covar[0] = sigma_noise**2
    maker = mass.FilterMaker(signal, npre, noise_covar, peak=Maxsignal)
    F5 = maker.compute_5lag()

    # Check filter's normalization
    f = F5.values
    verify_close(Maxsignal, f.dot(truncated_signal), rtol=1e-5, topic = "Filter normalization")

    # Check filter's variance 
    expected_dV = sigma_noise / n**0.5 * signal.max()/truncated_signal.std()
    verify_close(expected_dV, F5.variance**0.5, rtol=1e-5, topic="Expected variance")

    # Check filter's V/dV calculation
    fwhm_sigma_ratio = np.sqrt(8*np.log(2))
    expected_V_dV = Maxsignal / (expected_dV * fwhm_sigma_ratio)
    verify_close(expected_V_dV, F5.predicted_v_over_dv, rtol=1e-5, topic="Expected V/\u03b4v")
    print()

test_mass_5lag_filters(100, 1.0, 500)
test_mass_5lag_filters(400, 1.0, 500)
test_mass_5lag_filters(100, 1.0, 1000)
test_mass_5lag_filters(100, 2.0, 500)

