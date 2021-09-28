#date: 2021-09-28T17:03:20Z
#url: https://api.github.com/gists/efff01b2a75db215037fa191ac0c9ede
#owner: https://api.github.com/users/amotl

# Cedar Grove NAU7802 FeatherWing example
# 2021-01-07 v01 Cedar Grove Studios
# https://github.com/CedarGroveStudios/NAU7802_24-bit_ADC_FeatherWing/blob/main/examples/clue_scale/clue_scale_code.py

# Import library
# https://github.com/CedarGroveStudios/NAU7802_24-bit_ADC_FeatherWing/blob/main/code/cedargrove_nau7802.py
from cedargrove_nau7802 import NAU7802

# Configure ADC
DEFAULT_CHAN =   1  # Select load cell channel input; channel A=1, channel B=2
SAMPLE_AVG   = 100  # Number of sample values to average
MAX_GR       = 100  # Maximum (full-scale) display range in grams
MIN_GR       = ((MAX_GR // 5 ) * -1)  # Calculated minimum display value
DEFAULT_GAIN = 128  # Default gain for internal PGA

# Load cell dime-weight calibration ratio; 2.268 gm / ADC_raw_measurement
# Obtained emperically; individual load cell dependent
CALIB_RATIO = 100 / 215300  # 100g at gain x128 for load cell serial#4540-02

# Instantiate 24-bit load sensor ADC
nau7802 = NAU7802(board.I2C(), address=0x2A, active_channels=2)


def zero_channel():
    # Initiate internal calibration for current channel; return raw zero offset value
    # Use when scale is started, a new channel is selected, or to adjust for measurement drift
    # Remove weight and tare from load cell before executing
    print('channel %1d calibrate.INTERNAL: %5s'
          % (nau7802.channel, nau7802.calibrate('INTERNAL')))
    print('channel %1d calibrate.OFFSET:   %5s'
          % (nau7802.channel, nau7802.calibrate('OFFSET')))
    zero_offset = read(100)  # Read average of 100 samples to establish zero offset
    print('...channel zeroed')
    return zero_offset


def read(samples=100):
    # Read and average consecutive raw sample values; return average raw value
    sum = 0
    for i in range(0, samples):
        if nau7802.available:
            sum = sum + nau7802.read()
    return int(sum / samples)


def get_tare(value=None):
    # Measure and store tare weight; return raw, grams, and ounces values
    if value is None:
        # Read average of 100 samples and store raw tare offset
        tare_offset = read(100)
        tare_state = True
    else:
        # Set raw tare offset to zero and disable tare display
        tare_offset = 0
        tare_state  = False
    tare_gr_offset = round(tare_offset * CALIB_RATIO, 3)
    tare_oz_offset = round(tare_gr_offset * 0.03527, 4)
    return tare_offset, tare_gr_offset, tare_oz_offset


def main():

    print('    enable NAU7802 digital and analog power: %5s' % (nau7802.enable(True)))

    nau7802.gain    = DEFAULT_GAIN        # Use default gain
    nau7802.channel = DEFAULT_CHAN        # Set to default channel
    zero = zero_channel()                 # Calibrate and get raw zero offset value
    tare, tare_gr, tare_oz = get_tare(0)  # Disable tare subtraction and display


    ### Main loop: Read sample, move bubble, and display values
    #     Monitor Zeroing and Tare buttons
    while True:
        value   = read(SAMPLE_AVG)
        mass_gr = round((value - zero - tare) * CALIB_RATIO, 1)
        mass_oz = round(mass_gr * 0.03527, 2)

        mass_gr_value.text = '%5.1f' % (mass_gr)
        mass_oz_value.text = '%2.2f' % (mass_oz)
        tare_gr_value.text = '%6.1f' % (tare_gr)
        tare_oz_value.text = '%2.2f' % (tare_oz)

        print('(%+5.1f, %+2.2f)' % (mass_gr, mass_oz))
        # print('raw value:', value, hex(value))


if __name__ == "__main__":
    main()
