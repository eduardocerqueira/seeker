#date: 2022-05-04T17:20:08Z
#url: https://api.github.com/gists/71c370fdf1ed74e48d5bc5c928ef0f81
#owner: https://api.github.com/users/marcusmueller

"""
An activity detector that executes external programs and has a cooldown period
"""

import numpy as np
from gnuradio import gr
import shlex
import subprocess


class blk(gr.sync_block
          ):  # other base classes are basic_block, decim_block, interp_block
    """A simple command executor with a cooldown period"""
    def __init__(self,
                 command="touch /tmp/it_happened",
                 sampling_rate=1.0,
                 cooldown=1.0):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name='Embedded Python Block',  # will show up in GRC
            in_sig=[np.float32],
            out_sig=[])
        self.cooldown_samples = int(cooldown * sampling_rate)
        self.cooldown_left = 0
        if not command:
            self.command = None
        else:
            self.command = shlex.split(command)

        print(f"Set up for a cooldown of {self.cooldown_samples:d} samples")

    def work(self, input_items, output_items):
        # Check whether we're still cooling down from the last activation
        if self.cooldown_left > 0:
            # OK, then just consume as much input as we have, up to the full remainder of the cooldown period
            to_consume = min(self.cooldown_left, len(input_items[0]))
            self.cooldown_left -= to_consume
            return to_consume

        # we don't have any cooldown left, so let's look for a value > 0.5
        index = np.argwhere(input_items[0] > 0.5)
        if len(index) > 0:
            first = index[0]
            self.cooldown_left = self.cooldown_samples
            print("activated")
            if self.command:
                subprocess.Popen(self.command)
            return first
        # We're neither in cooldown, nor have we detected any high values – just consume all data
        return len(input_items[0])
