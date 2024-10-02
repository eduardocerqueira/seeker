#date: 2024-10-02T17:03:25Z
#url: https://api.github.com/gists/165f5bfa0aebbaa515b820a520dfd19b
#owner: https://api.github.com/users/justinturpin

"""
Wifi IR remote.
"""

import time
from machine import PWM, SPI, Pin, reset
import rp2
import sys
import network
import socket

WIFI_SSID = ""
WIFI_PASSWORD = "**********"


p_in = Pin(26, Pin.IN, Pin.PULL_UP)
p_in.irq(None)

p_out = Pin(16, Pin.OUT)
p_out.off()


@rp2.asm_pio(
    set_init=rp2.PIO.OUT_LOW,
    out_init=rp2.PIO.OUT_LOW,
    out_shiftdir=rp2.PIO.SHIFT_RIGHT,
    autopull=True,
    pull_thresh=32,
)
def pulsetrain():
    # Data must be in the form of [high time, low time, high time, low time, ...]
    # Subtract 2 from all values in the data queue because, for timing's sake,
    # there will be 2 rise and fall cycles.
    # Start the rising edge to start. 
    set(pins, 1) [2]
    out(x, 32)
    label("pwm_on")
    set(pins, 0) [3]
    set(pins, 1) [2]
    jmp(x_dec, "pwm_on")
    
    # Wait 2 cycles here to line up with what the HIGH logic does above
    # (otherwise it would just be set(pins, 0)[6]
    # Also wait an extra 4 cycles to finish the falling edge of the PWM
    set(pins, 0) [18]
    out(x, 32)
    label("pwm_off")
    nop() [6]
    jmp(x_dec, "pwm_off")


@rp2.asm_pio(
    in_shiftdir=rp2.PIO.SHIFT_LEFT,
)
def pulseinput():
    # Two instructions to read 1 signal, so signal rate is PIO freq/2
    in_(pins, 1)
    push(block)


def do_send_sm(signal: list[int]):
    # IR remotes send a PWM signal at 38khz. Set the PIO program rate to 4x that,
    # because it needs to do some work in between pulses
    sm_freq = 38_000 * 4
    data_freq = 19000
    data_multiplier = round(sm_freq / data_freq / 4)
    expected_time = 1000 * sum(signals.projector_power) / data_freq
    signal_sum = sum(signals.projector_power)
    
    print(f"frequency={sm_freq}")
    print("Num signals:", len(signal), signal_sum)
    print(f"Each signal should take {sm_freq / data_freq} cycles")
    print(f"Expected to finish in {expected_time}")
    
    sm = rp2.StateMachine(0, pulsetrain, freq=sm_freq, out_base=p_out, set_base=p_out)
    start_time = time.ticks_us()
    sm.active(1)
    
    for v in signal:
        sm.put(v - 2)
        
    while sm.tx_fifo():
        pass
    
    duration = (time.ticks_us() - start_time) / 1000
    cycle_diff = sm_freq * (duration - expected_time)
    
    print("Finished in", duration)
    print("Cycle difference of", cycle_diff / signal_sum / 1000)
    

def resolve_signal(signal: bytearray, length: int, freq: int) -> tuple[list[int], int]:
    """
    Convert the signal into a list of pulse lengths. Assume first pulse value is 1
    """
    prev_v = 1
    prev_t = 0
    pulses = []
    for t in range(length):
        v = signal[t]
        if v != prev_v:
            pulses.append(t - prev_t)
            prev_t = t
        prev_v = v
    pulses.append(t - prev_t + 1)
    assert sum(pulses) == length, f"Error, got {sum(pulses)}, expected {length}"
    return (pulses, freq)


def do_read():
    # Create signal buffer. First signal will always be 1 because that's
    # what we wait for
    signal = bytearray(20_000)
    signal[0] = 1
    i = 1
    off_count = 0
    
    # Read at 38khz. This is overkill but makes the timing more convenient when
    # it's time to reproduce the signal
    sm_freq = 38_000
    sm = rp2.StateMachine(0, pulseinput, freq=sm_freq, in_base=p_in)
    sm.active(1)
    
    # We are pulling HIGH, so wait for the signal to go low, which
    # means a pulse has begun
    print("Waiting for signal to begin...")
    
    while sm.get() == 1:
        pass

    while True:
        # This loop needs to be as performant as possible, so keep logic
        # to a bare minimum.
        # Invert pulled-HIGH signal value to get true value
        value = 1 - sm.get()
        signal[i] = value
        
        if value:
            off_count = 0
        else:
            off_count += 1
            
            if off_count >= 400:
                # Break when we detect a long series of zeroes.
                # Subtract 398, which will keep 1 zero at the end for padding.
                i -= 398
                break
        
        i += 1
    
    data_freq = sm_freq // 2
    print(f"Captured {i} signals in {i * 1000 / data_freq}ms")
    pulses, new_freq = resolve_signal(signal, i, sm_freq // 2)
    print(f"pulses = {pulses}")
    print(f"data rate = {new_freq}")
    

def do_web():
    # TODO: this is not remotely finished
    # Set up state machine
    sm_freq = 38_000 * 4
    sm = rp2.StateMachine(1, pulsetrain, freq=sm_freq, out_base=p_out, set_base=p_out)
    sm.active(1)
    
    #Connect to WLAN
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)
    
    while wlan.isconnected() == False:
        print('Waiting for connection...')
        time.sleep(4)
        
    print("Connected at", wlan.ifconfig()[0])
    
    address = ("0.0.0.0", 80)
    connection = socket.socket()
    connection.bind(address)
    connection.listen(1)
    
    while True:
        sm.restart()
        client = connection.accept()[0]
        request = client.recv(1024)
        request = str(request)
        print(request)
        path = request.split()[1]
        for v in signals.projector_power:
            sm.put(v - 2)
        while sm.tx_fifo():
            pass
        html = "HTTP/1.1 200 OK\r\nConnection: Closed\r\nContent-Type: text/plain\r\n\r\nhello!"
        client.send(html)
        client.close()
    
    time.sleep(5)
    
    reset()
    
do_read()
# do_send_sm(signal)
# do_web()
