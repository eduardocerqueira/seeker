#date: 2026-01-19T17:18:55Z
#url: https://api.github.com/gists/b63177a8a36201e9012074c20f233c78
#owner: https://api.github.com/users/tiation

#!/usr/bin/env python

class TermSequence:
    '''Reference: https://stackoverflow.com/a/33206814/1006369
    '''
    sequence = []

    # standard 4-bit color codes
    black          = 30
    red            = 31
    green          = 32
    yellow         = 33
    blue           = 34
    magenta        = 35
    cyan           = 36
    white          = 37
    bright_black   = 90
    bright_red     = 91
    bright_green   = 92
    bright_yellow  = 93
    bright_blue    = 94
    bright_magenta = 95
    bright_cyan    = 96
    bright_white   = 97

    @staticmethod
    def ESC(code, closed=True):
        return f'\033[{code}{"m" if closed else ""}'

    @property
    def clear(self):
        return self.append(TermSequence.ESC('H', False)).append(TermSequence.ESC('2J', False))

    @property
    def save_cur(self):
        return self.append(TermSequence.ESC('s', False))

    @property
    def restore_cur(self):
        return self.append(TermSequence.ESC('u', False))

    @property
    def erase_eol(self):
        return self.append(TermSequence.ESC('K', False))

    @property
    def erase_line(self):
        return self.append(TermSequence.ESC('2K', False))

    @property
    def cur_up(self, length):
        return self.append(TermSequence.ESC(f'{length}A', False))

    @property
    def cur_down(self, length):
        return self.append(TermSequence.ESC(f'{length}B', False))

    @property
    def cur_right(self, length):
        return self.append(TermSequence.ESC(f'{length}C', False))

    @property
    def cur_left(self, length):
        return self.append(TermSequence.ESC(f'{length}D', False))

    def cur_pos(self, line, col):
        return self.append(TermSequence.ESC(f'{line};{col}H', False))

    @property
    def reset(self):
        return self.append(TermSequence.ESC(0))

    @property
    def bold(self):
        return self.append(TermSequence.ESC(1))

    @property
    def dim(self):
        return self.append(TermSequence.ESC(2))

    @property
    def italic(self):
        return self.append(TermSequence.ESC(3))

    @property
    def underline(self):
        return self.append(TermSequence.ESC(4))

    @property
    def slow_blink(self):
        return self.append(TermSequence.ESC(5))

    @property
    def rapid_blink(self):
        return self.append(TermSequence.ESC(6))

    @property
    def reverse(self):
        return self.append(TermSequence.ESC(7))

    @property
    def hide(self):
        return self.append(TermSequence.ESC(8))

    @property
    def crossed(self):
        return self.append(TermSequence.ESC(9))

    @property
    def bold_off(self):
        return self.append(TermSequence.ESC(22))

    @property
    def normal(self):
        return self.append(TermSequence.ESC(22))

    @property
    def italic_off(self):
        return self.append(TermSequence.ESC(23))

    @property
    def underline_off(self):
        return self.append(TermSequence.ESC(24))

    @property
    def blink_off(self):
        return self.append(TermSequence.ESC(25))

    @property
    def reverse_off(self):
        return self.append(TermSequence.ESC(27))

    @property
    def show(self):
        return self.append(TermSequence.ESC(28))

    @property
    def crossed_off(self):
        return self.append(TermSequence.ESC(29))

    def fg_16(self, color):
        '''4-bit foreground color codes
        '''
        assert 30 <= color <= 37 or 90 <= color <= 97
        return self.append(TermSequence.ESC(color))

    def bg_16(self, color):
        '''4-bit color codes
        '''
        assert 40 <= color <= 47 or 100 <= color <= 107
        return self.append(TermSequence.ESC(color))

    def fg_256(self, color):
        '''8-bit foreground color code
        '''
        assert 0 <= color <= 256
        return self.append(TermSequence.ESC(f'38;5;{color}'))

    def bg_256(self, color):
        '''8-bit background color code
        '''
        assert 0 <= color <= 256
        return self.append(TermSequence.ESC(f'48;5;{color}'))

    def fg_rgb(self, r, g, b):
        '''RGB foreground color
        '''
        assert 0 <= r <= 256 or 0 <= b <= 256 or 0 <= b <= 256
        return self.append(TermSequence.ESC(f'38;2;{r};{g};{b}'))

    def bg_rgb(self, r, g, b):
        '''RGB background color
        '''
        assert 0 <= r <= 256 or 0 <= g <= 256 or 0 <= b <= 256
        return self.append(TermSequence.ESC(f'48;2;{r};{g};{b}'))

    @property
    def fg_default(self):
        return self.append(TermSequence.ESC(39))

    @property
    def bg_default(self):
        return self.append(TermSequence.ESC(49))

    @property
    def overline(self):
        return self.append(TermSequence.ESC(53))

    @property
    def overline_off(self):
        return self.append(TermSequence.ESC(55))

    def append(self, v):
        self.sequence.append(v)
        return self

    def __str__(self):
        return ''.join(self.sequence)

    def text(self, text):
        return self.append(text)

    def print(self, *vargs, **kvargs):
        print(self, end='')

    @property
    def empty(self):
        self.sequence.clear()
        return self


def demo():
    tp = TermSequence()

    tp.clear.print()
    tp.save_cur
    tp.cur_pos(5, 50).erase_line.text('This text is printed in coordinates (line=5, col=50).')
    tp.restore_cur

    print(tp)
    print(tp.empty.bold.text('Bold on').bold_off.text(' - Bold off').reset)
    print(tp.empty.dim.text('Dim on').normal.text(' - Dim off').reset)
    print(tp.empty.italic.text('Italic on').italic_off.text(' - Italic off').reset)
    print(tp.empty.overline.text('Overline on').overline_off.text(' - Overline off').reset)
    print(tp.empty.underline.text('Underline on').underline_off.text(' - Underline off').reset)
    print(tp.empty.slow_blink.text('Slow Blink').blink_off.text(' - Blink off').reset)
    print(tp.empty.rapid_blink.text('Rapid Blink').blink_off.text(' - Blink off').reset)
    print(tp.empty.reverse.text('Reverse on').reverse_off.text(' - Reverse off').reset)
    print(tp.empty.hide.text('Hide').show.text(' - Show').reset)
    print(tp.empty.crossed.text('Crossed on').crossed_off.text(' - Crossed off').reset)

    print()
    print("Color palete: 16")
    for i in range(30, 37):
        print(tp.empty.fg_16(i).text(f'  Foreground Color {i}').reset, end='')
        print("\t", end='')
        print(tp.empty.bg_16(i+10).text(f'  Background Color {i}').reset)

    print()
    print("Color palete: 256")
    for i in range(0, 255, 5):
        print(tp.empty.fg_256(i).text(f'  Foreground Color {i}').reset, end='')
        print("\t", end='')
        print(tp.empty.bg_256(i).text(f'  Background Color {i}').reset)

    print()
    print("Color palete: RBG")
    for r in range(0, 255, 50):
        for g in range(0, 255, 50):
            for b in range(0, 255, 50):
                print(tp.empty.fg_rgb(r, g, b).text(f'  Foreground Color {r:03}, {g:03}, {b:03}').reset, end='')
                print("\t", end='')
                print(tp.empty.bg_rgb(r, g, b).text(f'  Background Color {r:03}, {g:03}, {b:03}').reset)

if __name__ == '__main__':
    demo()