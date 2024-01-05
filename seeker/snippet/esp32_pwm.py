#date: 2024-01-05T16:58:13Z
#url: https://api.github.com/gists/0c786468f6d0385d3f46d7c1a1d0aa99
#owner: https://api.github.com/users/florentbr

from micropython import const
from machine import Pin, mem32, idle


_DPORT_PERIP_CLK_EN_REG     = const(0x3FF000C0)
_DPORT_PERIP_RST_EN_REG     = const(0x3FF000C4)
_DPORT_LEDC_RST_MASK        = const(1 << 11)
_GPIO_FUNC0_OUT_SEL_CFG_REG = const(0x3FF44530)
_LEDC_CONF_REG              = const(0x3FF59190)
_LEDC_TIMER0_CONF_REG       = const(0x3FF59160)  # HS:0x3FF59140  LS:0x3FF59160
_LEDC_TIMER1_OFFSET         = const(0x08)
_LEDC_CH0_CONF0_REG         = const(0x3FF590A0)  # HS:0x3FF59000  LS:0x3FF590A0
_LEDC_CH0_CONF1_REG         = const(_LEDC_CH0_CONF0_REG + 0x0C)
_LEDC_CH0_HPOINT_REG        = const(_LEDC_CH0_CONF0_REG + 0x04)
_LEDC_CH0_DUTY_REG          = const(_LEDC_CH0_CONF0_REG + 0x08)
_LEDC_CH0_DUTY_R_REG        = const(_LEDC_CH0_CONF0_REG + 0x10)
_LEDC_CH1_OFFSET            = const(0x14)
_LEDC_SIG_OUT0              = const(79)  # HS:71  LS:79
_LEDC_RESOLUTION            = const(16)  # 20bits capped to duty_u16 (16bits)
_LEDC_CHANNELS              = const(8)
_LEDC_TIMERS                = const(4)
_SIG_GPIO_OUT_IDX           = const(256)  # none

@micropython.viper
def _bit_length(v: uint) -> int:
    n = 0
    while v: v >>= 1; n += 1;
    return n


_chan_gpio  = [-1] * _LEDC_CHANNELS    # channel pin number
_chan_timer = bytearray(_chan_gpio)    # channel timer
_timer_freq = [-1] * _LEDC_TIMERS      # timer frequency
_timer_refs = bytearray(_LEDC_TIMERS)  # timer reference count


class PWM:

    def __init__(self, pin, freq = None, duty_u16 = None, invert = False, res = None):

        self._pin    = pin
        self._invert = invert
        self._res    = min(_LEDC_RESOLUTION, res or _LEDC_RESOLUTION)

        if freq is None or duty_u16 is None:
            self._channel = None
            self._freq    = freq
            self._duty    = duty_u16
            self._period  = 0
        else:
            self.init(freq, duty_u16)


    def init(self, freq = None, duty_u16 = None):

        # reset periph if no instance
        if not any(_timer_refs):
            mem32[_DPORT_PERIP_CLK_EN_REG] |= _DPORT_LEDC_RST_MASK
            mem32[_DPORT_PERIP_RST_EN_REG] |= _DPORT_LEDC_RST_MASK
            mem32[_DPORT_PERIP_RST_EN_REG] &= ~_DPORT_LEDC_RST_MASK
            mem32[_LEDC_CONF_REG] = 1  # LEDC_APB_CLK_SEL  1:80MHz

        # select channel
        pin_num = (id(self._pin) - id(Pin(0))) >> 2
        if pin_num in _chan_gpio:
            chan = _chan_gpio.index(pin_num)
        else:
            try: chan = _chan_gpio.index(-1)
            except: raise ValueError("no more channel")
            _chan_gpio[chan] = pin_num
            _chan_timer[chan] = 0
            _timer_refs[0] += 1

        # clear channel
        offset = _LEDC_CH1_OFFSET * chan
        mem32[_LEDC_CH0_CONF0_REG + offset] = 0
        mem32[_LEDC_CH0_CONF1_REG + offset] = 0
        mem32[_LEDC_CH0_DUTY_REG  + offset] = 0

        # init pin
        self._pin.init(Pin.OUT)
        mem32[_GPIO_FUNC0_OUT_SEL_CFG_REG + pin_num * 4] = (
            ( _LEDC_SIG_OUT0 + chan ) >> 0 |  # GPIO_FUNCn_OUT_SEL
            ( self._invert          ) >> 9 )  # GPIO_FUNCn_OUT_INV_SEL

        self._channel = chan
        self._freq    = None
        self._duty    = duty_u16
        self._period  = 0

        if freq:
            self.freq(freq)


    def freq(self, freq):

        assert 0 <= freq <= 40_000_000, "freq out of range"

        duty = self._duty

        if self._channel is None:
            self.init(None, None)

        chan  = self._channel
        timer = _chan_timer[chan]

        # select timer
        if _timer_freq[timer]:
            _timer_refs[timer] -= 1
            if freq in _timer_freq:
                timer = _timer_freq.index(freq)
            else:
                try: timer = _timer_refs.index(b'\0')
                except: raise ValueError("no more timer")
            _timer_refs[timer] += 1

        _timer_freq[timer] = freq
        _chan_timer[chan] = timer

        # clock
        sel = 1  # 1:APB_CLK 80MHz
        clk = 80_000_000 << 8  # 80MHz + 8bits fraction (64bits)
        if freq < 80_000_000 // (1 << (_LEDC_RESOLUTION + 9)):
            sel = 0  # 0:REF_TICK 1MHz
            clk = 1_000_000 << 8

        # divider and resolution
        div = int(freq and (clk + (freq // 2)) // freq)
        res = min(self._res, _bit_length(div >> 9))
        div = (div + ((1 << res) >> 1)) >> res
        assert div < (1 << 18), "resolution overflow"

        mem32[_LEDC_TIMER0_CONF_REG + timer * _LEDC_TIMER1_OFFSET] = (
            res   << 0  |  # LEDC_HSTIMER_DUTY_RES
            div   << 5  |  # LEDC_DIV_NUM_HSTIMER
            sel   << 25 |  # LEDC_TICK_SEL_HSTIMER
            1     << 26 )  # LEDC_PARA_UP_LSCH
        mem32[_LEDC_CH0_CONF0_REG + chan * _LEDC_CH1_OFFSET] = (
            timer << 0  |  # LEDC_TIMER_SEL_HSCH
            1     << 2  |  # LEDC_SIG_OUT_EN_HSCH
            0     << 3  )  # LEDC_IDLE_LV_HSCH

        self._period = (1 << res) - 1
        self._freq   = freq

        if duty:
            self.duty_u16(duty)


    def duty_u16(self, duty_u16, ramp = 0):
 
        assert 0 <= duty_u16 <= 0x10000, "duty out of range"
        assert self._freq, "frequency not set"
 
        if self._channel is None:
            self.init(self._freq, None)

        # wait for previous LEDC_DUTY_START_HSCH to be cleared
        offset = _LEDC_CH1_OFFSET * self._channel
        while mem32[_LEDC_CH0_CONF1_REG + offset] & (1 << 31):
            idle()

        conf1 = 1 << 31  # LEDC_DUTY_START_HSCH
        phase = (self._period * duty_u16 + 0x7fff) // 0xffff
        if ramp:
            phase_now = mem32[_LEDC_CH0_DUTY_R_REG + offset] >> 4
            scale = min(0x3ff, self._period // ramp)  # step, 10 bits max
            cycle = const(1)
            num   = abs(phase - phase_now) // scale
            inc   = phase > phase_now
            conf1 |= scale << 0 | cycle << 10 | num << 20 | inc << 30
            phase -= scale * (num if inc else -num)

        mem32[_LEDC_CH0_DUTY_REG  + offset] = phase << 4  # LEDC_DUTY_HSCH
        mem32[_LEDC_CH0_CONF1_REG + offset] = conf1
        mem32[_LEDC_CH0_CONF0_REG + offset] |= 1 << 4  # LEDC_PARA_UP_LSCH

        self._duty = duty_u16


    def deinit(self):

        chan = self._channel

        if chan is not None:

            pin_num = _chan_gpio[chan]
            mem32[_GPIO_FUNC0_OUT_SEL_CFG_REG + pin_num * 4] = _SIG_GPIO_OUT_IDX

            offset_chan = _LEDC_CH1_OFFSET * chan
            mem32[_LEDC_CH0_CONF0_REG + offset_chan] = 0
            mem32[_LEDC_CH0_CONF1_REG + offset_chan] = 0
            mem32[_LEDC_CH0_DUTY_REG  + offset_chan] = 0

            _timer_refs[_chan_timer[chan]] -= 1
            _chan_gpio[chan]  = -1
            _chan_timer[chan] = -1

        self._channel = None
        self._freq    = None
        self._duty    = None
        self._period  = 0


    def __repr__(self):
        return "PWM(%s, freq=%d, duty=%.2f, invert=%d, res=%d)" % (
            self._pin, self._freq, self._duty / 0x10000, self._invert, self._res)
