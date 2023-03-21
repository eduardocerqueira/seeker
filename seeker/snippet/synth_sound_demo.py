#date: 2023-03-21T16:54:02Z
#url: https://api.github.com/gists/b1ca0e61a9ea46885db6e8c74243838b
#owner: https://api.github.com/users/zvodd

import pygame
import pygame_gui
import math
import pyaudio
import struct

SCREEN_X = 800
SCREEN_Y = 600

SCOPE_DIMS = (SCREEN_X//4, SCREEN_Y // 4)

SCOPE_Y_SCALE = SCOPE_DIMS[1]
SCOPE_X_SCALE = 2

SAMPLE_RATE = 44100
BUFFER_SIZE = 1024

FREQ_A4 = 440

def generate_wave(freq, sample_rate, length, offset = 0, scale = 0.1, type="sin"):
    #eliminate popping by offsetting the buffer by a wave period
    offset = offset % (sample_rate / freq)
    # sample a wave of freq by a sample_rate for length samples; 
    # Same as plotting y = f(x), whilst specifiying the x values with range and steps
    if type == "sin":
        for i in range(length):
            yield scale * math.sin(math.tau * freq * (i + offset) / sample_rate)
    elif type == "saw":
        for i in range(length):
            # just copied and modified from above, scale seems off?
            yield scale * (math.pi/2 - math.fmod((math.tau * freq * (i + offset) / sample_rate), math.pi))
    elif type == "square":
        for i in range(length):
            yield scale * math.copysign(1, math.sin(math.tau * freq * (i + offset) / sample_rate))
    elif type == "triangle":
        for i in range(length):
            yield scale * (math.asin(math.sin(math.tau * freq * (i + offset) / sample_rate)) * 2 / math.pi)
    elif type == "sigmoid":
        for i in range(length):
            yield scale * (math.exp(math.sin(math.tau * freq * (i + offset) / sample_rate))-1.5)


def mix_wave(wave_a, wave_b, ratio = 0.5):
    fac_a = 0 + ratio
    fac_b = 1 - ratio
    for i, sample in enumerate(wave_a):
        yield (sample * fac_a) + (wave_b[i] * fac_b)


def emi_bandpass(wave, freq, sample_rate, q=1, fltregs = [0,0,0,0]):
    """ Basic emi feedback filter, need to keep registers between calls to avoid popping"""
    # Calculate the angular frequencies
    w0 = 2 * math.pi * freq / sample_rate
    alpha = math.sin(w0) / (2 * q)

    # Calculate the coefficients
    a0 = 1 + alpha
    a1 = -2 * math.cos(w0)
    a2 = 1 - alpha
    b0 = alpha
    b1 = 0
    b2 = -alpha

    # Initialize the delay registers
    x1, x2 = fltregs[0], fltregs[1]
    y1, y2 = fltregs[2], fltregs[3]

    # Apply the filter to the waveform
    for x in wave:
        y = (b0/a0)*x + (b1/a0)*x1 + (b2/a0)*x2 - (a1/a0)*y1 - (a2/a0)*y2
        yield y
        x2, x1 = x1, x
        y2, y1 = y1, y
        fltregs[0], fltregs[1] = x1, x2
        fltregs[2], fltregs[3] = y1, y2 




def main():
    pygame.init()
    pygame.display.set_caption("Wave")
    clock = pygame.time.Clock()

    background = pygame.Surface((SCREEN_X, SCREEN_Y))
    background.fill(pygame.Color('#000000'))
    window_surface = pygame.display.set_mode((SCREEN_X, SCREEN_Y))

    nwavs = 4
    signal_labels = ["Signal A", "Signal B", "Mixed", "Filtered"]
    wave_types = ["saw", "sin", "square", "triangle", "sigmoid"]
    wave_1_type = 0
    wave_2_type = 1
    wavsurfs = [pygame.Surface(SCOPE_DIMS) for _ in range(nwavs)]
    waves = [[] for _ in range(nwavs)]

    mixfac = 0.5
    wavoff = 0 # running total of samples generated; i.e. buffer_SIZE * calls to gen_waves
    select_wav = 2
    
    filter_freq = 2000
    filter_q = .5
    fltregs = [0, 0, 0, 0] # keep filter state between calls to emi_bandpass
    
    # generate the next wave buffer, first call is not offset to prevent popping
    def gen_waves(offset = True):
        nonlocal wavoff
        waves[0] = [*generate_wave(FREQ_A4, SAMPLE_RATE, BUFFER_SIZE, offset=wavoff, type=wave_types[wave_1_type])]
        waves[1] = [*generate_wave(FREQ_A4 * math.pow(2, 4/12), SAMPLE_RATE, BUFFER_SIZE, offset=wavoff, type=wave_types[wave_2_type])]
        waves[2] = [*mix_wave(waves[0], waves[1], mixfac)]
        waves[3] = [*emi_bandpass(waves[2], filter_freq, SAMPLE_RATE, q=filter_q, fltregs=fltregs)]
        if offset:
            wavoff = wavoff + BUFFER_SIZE

    gen_waves(offset=False)
    
    # sample wave functions and send selected wave buffer to audio output
    # this is called by pyaudio at a rate of SAMPLE_RATE / buffer_size (e.g. 44100 / 1024 = 43.07 times per second) or every 23.22ms
    def audio_callback(in_data, frame_count, time_info, status):
        gen_waves()
        data = waves[select_wav][:frame_count]
        data = struct.pack("%df" % frame_count, *data)
        return (data, pyaudio.paContinue)

    #setup audio device
    au = pyaudio.PyAudio()
    stream = au.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    frames_per_buffer=BUFFER_SIZE,
                    stream_callback=audio_callback,
                    output=True,)

    #setup gui
    manager = pygame_gui.UIManager((SCREEN_X, SCREEN_Y))
    but_playpause = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((110 * 0, SCREEN_Y - 50), (100, 50)), text='Pause',manager=manager)
    but_output_select = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((110 * 1, SCREEN_Y - 50), (100, 50)), text=signal_labels[select_wav], manager=manager)
    # scrollb_mix = pygame_gui.elements.UIVerticalScrollBar(relative_rect=pygame.Rect(60, 30, 20, 300), visible_percentage=0.2,manager=manager)
    but_waveselect_1 = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(10, 5, 100, 20), text=wave_types[wave_1_type], manager=manager)
    but_waveselect_2 = pygame_gui.elements.UIButton(relative_rect=pygame.Rect(10, 340, 100, 20), text=wave_types[wave_2_type], manager=manager)

    window_surface.blit(background, (0, 0))
    # main loop
    while True:
        time_delta = clock.tick(60)/1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                stream.stop_stream()
                stream.close()
                au.terminate()
                pygame.quit()
                return
            
            # Why does the verticle scrollbar not have any events? Boooo!
            # if event.type == pygame_gui.UI_VERTICAL_SCROLL_BAR_MOVED:
            if event.type == pygame.MOUSEWHEEL:
                if event.y > 0:
                    mixfac = min(1, mixfac + 0.05)
                elif event.y < 0:
                    mixfac = max(0, mixfac - 0.05)
            
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == but_playpause:
                    if stream.is_active():
                        stream.stop_stream()
                        but_playpause.set_text('Play')
                    else:
                        stream.start_stream()
                        but_playpause.set_text('Pause')
                if event.ui_element == but_output_select:
                    select_wav = (select_wav + 1) % nwavs
                    but_output_select.set_text(signal_labels[select_wav])
                if event.ui_element == but_waveselect_1:
                    wave_1_type = (wave_1_type + 1) % len(wave_types)
                    but_waveselect_1.set_text(wave_types[wave_1_type])
                if event.ui_element == but_waveselect_2:
                    wave_2_type = (wave_2_type + 1) % len(wave_types)
                    but_waveselect_2.set_text(wave_types[wave_2_type])
            manager.process_events(event)
        


        # Draw scope backgrounds
        for i, surf in enumerate(wavsurfs):
            surf.fill(pygame.Color('#AAAAAA'))
            if select_wav == i:
                pygame.draw.rect(surf, pygame.Color('#FFDDAA'), (0, 0, SCOPE_DIMS[0], SCOPE_DIMS[1]), 10)
            pygame.draw.line(surf, (0, 0, 0), (0, SCOPE_Y_SCALE * 0.5), (SCOPE_DIMS[0], SCOPE_Y_SCALE * 0.5), 1)

        # Draw the waveforms to each surface
        pygame.draw.aalines(wavsurfs[0], (255, 50, 90), False, [(i/SCOPE_X_SCALE, SCOPE_Y_SCALE * (sample + 0.5)) for i, sample in enumerate(waves[0])])
        pygame.draw.aalines(wavsurfs[1], (30, 50, 255), False, [(i/SCOPE_X_SCALE, SCOPE_Y_SCALE * (sample + 0.5)) for i, sample in enumerate(waves[1])])
        pygame.draw.aalines(wavsurfs[2], (255, 50, 255), False, [(i/SCOPE_X_SCALE, SCOPE_Y_SCALE * (sample + 0.5)) for i, sample in enumerate(waves[2])])
        pygame.draw.aalines(wavsurfs[3], (20, 255, 20), False, [(i/SCOPE_X_SCALE, SCOPE_Y_SCALE * (sample + 0.5)) for i, sample in enumerate(waves[3])])

        #position and copy the surfaces to the main window
        window_surface.blit(wavsurfs[0], (80, 20))
        window_surface.blit(wavsurfs[1], (80, 40 + SCOPE_DIMS[1]))
        window_surface.blit(wavsurfs[2], (100 + SCOPE_DIMS[0], 30 + int(SCOPE_DIMS[1]/2)))
        window_surface.blit(wavsurfs[3], (110 + SCOPE_DIMS[0]*2, 30 + int(SCOPE_DIMS[1]/2)))

        manager.update(time_delta)
        manager.draw_ui(window_surface)
        
        pygame.display.update()


if __name__ == "__main__":
    main()


