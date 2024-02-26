#date: 2024-02-26T17:08:46Z
#url: https://api.github.com/gists/06c58eb45fdebfe1f245bf940a3f63f8
#owner: https://api.github.com/users/harshitakukreja

import argparse
import copy
import errno
import os

import numpy as np
import pyedflib as plib
from pyedflib import highlevel


def duplicate_channel_in_edf(input_file, channel_name, overwrite):

    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), input_file)

    # Read edf file
    signals, signal_headers, header = highlevel.read_edf(input_file)

    # Check if channel exists
    channel_index = -1

    for i in range(len(signal_headers)):
        if signal_headers[i]["label"] == channel_name:
            if channel_index != -1:
                raise Exception(f"Multiple channels found with the same label '{channel_name}'.")
            channel_index = i

    if channel_index == -1:
        raise Exception(f"Channel label '{channel_name}' does not exist in the EDF file.")

    duplicate_channel = copy.deepcopy(signal_headers[channel_index])
    duplicate_channel["label"] = "d" + channel_name
    signal_headers.append(duplicate_channel) # signal_headers is list of dictionaries

    duplicate_signal = copy.deepcopy(signals[channel_index])
    signals = np.vstack ((signals, duplicate_signal))  # signals is np array

    if overwrite:
        output_file = input_file
    else:
        base_file_name, extension = os.path.splitext(input_file)
        output_file = f"{base_file_name}_duplicate{extension}"

    is_duplicated = highlevel.write_edf(output_file, signals, signal_headers, header=header, file_type=-1)

    if is_duplicated:
        print(f"Channel '{channel_name}' duplicated successfully. New file: '{output_file}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Duplicates the user provided channel in an EDF file.')
    parser.add_argument('input_file', help='input EDF file path')
    parser.add_argument('channel_name', help='channel name to be duplicated')
    parser.add_argument('--overwrite', action="store_true", help='overwrite the original file')
    args = parser.parse_args()

    duplicate_channel_in_edf(args.input_file, args.channel_name, args.overwrite)