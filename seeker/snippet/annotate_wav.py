#date: 2025-03-07T17:10:41Z
#url: https://api.github.com/gists/e4ddcacbbbc18b10f4c0db08f7108b68
#owner: https://api.github.com/users/crolen

import sys
import os
import numpy as np
from pathlib import PurePath

import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from scipy.io import wavfile
import librosa
import sounddevice as sd


plt.rcParams["axes.xmargin"] = 0


class Annotate:
    def __init__(self, fname, out_path, LEN=3000, NFFT=256, NHOP=256, sr_out=16000):
        self.t_vec = []
        self.fname = fname
        self.out_path = out_path
        self.sr_out = sr_out
        self.y, sr_in = librosa.load(self.fname, mono=True)
        self.y = librosa.resample(self.y, orig_sr=sr_in, target_sr=self.sr_out)
        self.LEN = LEN
        self.NHOP = NHOP
        self.NFFT = NFFT

    def process(self):
        def save(event):
            if len(self.t_vec) > 0:
                n = 0
                strn = self.fname
                f = open(self.out_path + "/index.csv", "a")
                for ix in self.t_vec:
                    pp = PurePath(self.fname)
                    wavfile.write(
                        self.out_path + "/" + pp.stem + "_" + str(n) + ".wav",
                        self.sr_out,
                        self.y[ix : (ix + self.LEN)],
                    )
                    n += 1
                    strn += "," + str(ix)
                f.write(strn + "\n")
                f.close()
                print("     " + strn)
            plt.close()

        def play(event):
            sd.play(self.y, self.sr_out)
            # playsound(self.fname)

        def onPress(event):
            if event.inaxes == self.ax[0]:
                if event.button == event.button.RIGHT:
                    sd.play(
                        self.y[int(event.xdata) : (int(event.xdata) + self.LEN)],
                        self.sr_out,
                    )
                if event.key == " ":
                    # print(f"Sample: {int(event.xdata)} ")
                    x = np.arange(
                        int(event.xdata), int(event.xdata) + self.LEN, step=1, dtype=int
                    )
                    self.t_vec.append(x[0])
                    self.ax[0].plot(x, self.y[x], "y", alpha=0.9)
                    self.ax[0].figure.canvas.draw()

        # Setup figures
        fig, self.ax = plt.subplots(2, 1, figsize=(16, 8))
        fig.subplots_adjust(bottom=0.2)
        plt.suptitle(f"{self.fname}")

        ######################
        S = librosa.stft(self.y, n_fft=self.NFFT, hop_length=self.NHOP)
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
        # librosa.display.waveshow(y, sr=sr, axis = None, ax=ax[0][0])
        self.ax[0].plot(np.arange(len(self.y)), self.y, linewidth=0.5)
        self.ax[1].imshow(S_db, origin="lower", aspect="auto", interpolation="nearest")

        # y_harm, y_perc = librosa.effects.hpss(y)

        # Register callbacks for mouse events
        fig.canvas.mpl_connect("button_press_event", onPress)

        ax_save = fig.add_axes([0.8, 0.05, 0.1, 0.05])
        b_save = Button(ax_save, "Split & Save")
        b_save.on_clicked(save)

        ax_play = fig.add_axes([0.675, 0.05, 0.1, 0.05])
        b_play = Button(ax_play, "Play")
        b_play.on_clicked(play)

        ax_quit = fig.add_axes([0.55, 0.05, 0.1, 0.05])
        b_quit = Button(ax_quit, "Quit")
        b_quit.on_clicked(sys.exit)

        plt.show()


def main():
    # path = sys.argv[1]
    # fname = sys.argv[2]
    out_dir = "./SPLIT_TEST"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    fname = "aaa/mg.wav"
    ann = Annotate(fname, out_dir)
    ann.process()


if __name__ == "__main__":
    # Execute when the module is not initialized from an import statement.
    sys.exit(main())
