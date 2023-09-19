#date: 2023-09-19T17:05:42Z
#url: https://api.github.com/gists/d55547d8f334bf48e57453e31717b7e4
#owner: https://api.github.com/users/flisboac

from bisect import bisect_left
import math

r_mult_values_smd = [
    0.1, 1, 10, 100, 1000, 10000, 100000
]
r_base_values_smd =[
    10, 12, 15, 20, 22, 27, 30, 33,
    39, 47, 51, 56, 68, 75, 82, 91
]
r_all_values_smd = [
    base * mult
    for base in r_base_values_smd
    for mult in r_mult_values_smd
]

c_ceramic_mult_values_smd = [
    10 ** (-12), # pico
    10 ** ( -9), # nano
    10 ** ( -6), # micro
]
c_ceramic_base_values_smd =[
    # A-Z
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0,
    2.4, 1.7, 3.0, 3.3, 3.6, 3.9, 4.3, 4.7,
    5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1,
    # a-z
    2.5, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0,
]
c_ceramic_all_values_smd = [
    base * mult
    for base in c_ceramic_base_values_smd
    for mult in c_ceramic_mult_values_smd # For ceramic SMD, maybe?: if base * mult < 15000000
]


abs_r1_min = 100000
abs_r1_max = 2200000
abs_r2_min = 100
abs_r2_max = 200000


def find_closest_value(values, target_value, key=lambda item: item):
    values = sorted(values, key=key)
    pos = bisect_left(values, target_value, key=key)
    if pos == 0:
        return values[0]
    if pos == len(values):
        return values[-1]
    values_before = values[pos - 1]
    values_after = values[pos]
    if key(values_after) - target_value < target_value - key(values_before):
        return values_after
    else:
        return values_before

# vout = output voltage 
def calculate_vout(r1, r2):
    return 1.233 * (1 + (r1 / r2))

# cff = Feed-forward capacitor value
def calculate_c_ff(r1, fs):
    return 1 / (2 * math.pi * (fs / 20) * r1)

# i_peak = frequency for a specific load current
def calculate_i_peak(i_peak_nominal, vin, l):
    return i_peak_nominal + (vin / l) * 0.0000001

# fs_max = maximum switching frequency
def calculate_fs_max(vin_min, vout, vin, i_peak, l):
    return (vin_min * (vout - vin)) / (i_peak * l * vout)

# fs_load = frequency for a specific load current
def calculate_fs_load(vin, vout, vd_load, i_load, i_peak, l):
    return (2 * i_load * (vout - vin + vd_load)) / ((i_peak ** 2) * l)

def find_nearest_vout_config(vout):
    vout_configs = [
        (calculate_vout(r1, r2), r1, r2)
        for r1 in r_all_values_smd if abs_r1_min <= r1 <= abs_r1_max
        for r2 in r_all_values_smd if abs_r2_min <= r2 <= abs_r2_max
    ]
    return find_closest_value(vout_configs, vout, key=lambda i: i[0])

def generate_nearest_vout_configs(vout):
    vout_configs = [
        (calculate_vout(r1, r2), r1, r2)
        for r1 in r_all_values_smd if abs_r1_min <= r1 <= abs_r1_max
        for r2 in r_all_values_smd if abs_r2_min <= r2 <= abs_r2_max
    ]
    yield from sorted(vout_configs, key=lambda i: abs(i[0] - vout))

def find_nearest_c_ceramic_value(c_target):
    return min(c_ceramic_all_values_smd, key=lambda i: abs(c_target - i))

#generate_nearest_vout_configs(12)

l = 0.00001
"""Inductor's selected value, in Henry."""

i_peak_nominal = 0.4
"""Peak current as per the specs.
Use the following values:
- For the TPS61040-Q1: 0.4 (400mA)
- For the TPS61041-Q1: 0.25 (250mA)
"""

vin_target = 5
"""Target input voltage."""

vin_min = vin_target * 0.95
"""Minimum input voltage.
A good default (maybe?) is 5% less than target input voltage.
"""

vout_target = 12
"""Target output voltage."""

i_load = 0.4
"""Typical load current."""

vd_load = 0.34
"""Schottky diode's forward voltage @ load current.
This is just an estimate for 1N5819WS @ 400mA.
Check the datasheet for your selected diode for details.
"""

vout, r1, r2 = find_nearest_vout_config(vout_target)
i_peak = calculate_i_peak(i_peak_nominal, vin_target, l)
fs_max = calculate_fs_max(vin_min, vout, vin_target, i_peak, l)
fs_load = calculate_fs_load(vin_target, vout, vd_load, i_load, i_peak, l)
c_ff = calculate_c_ff(r1, fs_load)
c_ff_nearest = find_nearest_c_ceramic_value(c_ff)

print(dict(
    l=l,
    i_peak_nominal=i_peak_nominal,
    vin_target=vin_target,
    vin_min=vin_min,
    vout_target=vout_target,
    i_load=i_load,
    vd_load=vd_load,
    vout=vout,
    r1=r1,
    r2=r2,
    i_peak=i_peak,
    fs_max=fs_max,
    fs_load=fs_load,
    c_ff=c_ff,
    c_ff_nearest=c_ff_nearest,
))
