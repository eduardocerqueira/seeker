#date: 2024-06-17T17:10:17Z
#url: https://api.github.com/gists/dbf0b7d3fc383fa3384ef6f797df20dd
#owner: https://api.github.com/users/dm4

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# https://github.com/metabrainz/picard/blob/master/picard/disc/utils.py
# https://github.com/metabrainz/picard/blob/master/picard/disc/eaclog.py
# https://github.com/metabrainz/picard/blob/master/picard/util/__init__.py
#
# fix-header: nolicense
# MIT License
#
# Copyright(c) 2018 Konstantin Mochalov
# Copyright(c) 2022 Philipp Wolfer
# Copyright(c) 2022 Jeffrey Bosboom
#
# Original code from https://gist.github.com/kolen/765526
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

try:
    from charset_normalizer import detect
except ImportError:
    try:
        from chardet import detect
    except ImportError:
        detect = None

import re
from collections import namedtuple


PREGAP_LENGTH = 150
DATA_TRACK_GAP = 11400


TocEntry = namedtuple('TocEntry', 'number start_sector end_sector')


ENCODING_BOMS = {
    b'\xff\xfe\x00\x00': 'utf-32-le',
    b'\x00\x00\xfe\xff': 'utf-32-be',
    b'\xef\xbb\xbf': 'utf-8-sig',
    b'\xff\xfe': 'utf-16-le',
    b'\xfe\xff': 'utf-16-be',
}


class NotSupportedTOCError(Exception):
    pass


def detect_file_encoding(path, max_bytes_to_read=1024*256):
    """Attempts to guess the unicode encoding of a file based on the BOM, and
    depending on avalibility, using a charset detection method.

    Assumes UTF-8 by default if no other encoding is detected.

    Args:
        path: The path to the file
        max_bytes_to_read: Maximum bytes to read from the file during encoding
        detection.

    Returns: The encoding as a string, e.g. "utf-16-le" or "utf-8"
    """
    with open(path, 'rb') as f:
        first_bytes = f.read(4)
        for bom, encoding in ENCODING_BOMS.items():
            if first_bytes.startswith(bom):
                return encoding

        if detect is None:
            return 'utf-8'

        f.seek(0)
        result = detect(f.read(max_bytes_to_read))
        if result['encoding'] is None:
            log.warning("Couldn't detect encoding for file %r", path)
            encoding = 'utf-8'
        elif result['encoding'].lower() == 'ascii':
            # Treat ASCII as UTF-8 (an ASCII document is also valid UTF-8)
            encoding = 'utf-8'
        else:
            encoding = result['encoding'].lower()

        return encoding


def calculate_mb_toc_numbers(toc):
    """
    Take iterator of TOC entries, return a tuple of numbers for MusicBrainz disc id

    Each entry is a TocEntry namedtuple with the following fields:
    - number: track number
    - start_sector: start sector of the track
    - end_sector: end sector of the track
    """
    toc = tuple(toc)
    toc = _remove_data_track(toc)
    num_tracks = len(toc)
    if not num_tracks:
        raise NotSupportedTOCError("Empty track list")

    expected_tracknums = tuple(range(1, num_tracks+1))
    tracknums = tuple(e.number for e in toc)
    if expected_tracknums != tracknums:
        raise NotSupportedTOCError(f"Non-standard track number sequence: {tracknums}")

    leadout_offset = toc[-1].end_sector + PREGAP_LENGTH + 1
    offsets = tuple(e.start_sector + PREGAP_LENGTH for e in toc)
    return (1, num_tracks, leadout_offset) + offsets


def _remove_data_track(toc):
    if len(toc) > 1:
        last_track_gap = toc[-1].start_sector - toc[-2].end_sector
        if last_track_gap == DATA_TRACK_GAP + 1:
            toc = toc[:-1]
    return toc

RE_TOC_TABLE_HEADER = re.compile(r""" \s*
    \s*.+\s+ \| # track
    \s+.+\s+ \| # start
    \s+.+\s+ \| # length
    \s+.+\s+ \| # start sector
    \s+.+\s*$   # end sector
    """, re.VERBOSE)

RE_TOC_TABLE_LINE = re.compile(r"""
    \s*
    (?P<num>\d+)
    \s*\|\s*
    (?P<start_time>[0-9:.]+)
    \s*\|\s*
    (?P<length_time>[0-9:.]+)
    \s*\|\s*
    (?P<start_sector>\d+)
    \s*\|\s*
    (?P<end_sector>\d+)
    \s*$""", re.VERBOSE)


def filter_toc_entries(lines):
    """
    Take iterator of lines, return iterator of toc entries
    """

    # Search the TOC table header
    for line in lines:
        # to allow internationalized EAC output where column headings
        # may differ
        if RE_TOC_TABLE_HEADER.match(line):
            # Skip over the table header separator
            next(lines)
            break

    for line in lines:
        m = RE_TOC_TABLE_LINE.search(line)
        if not m:
            break
        yield TocEntry(int(m['num']), int(m['start_sector']), int(m['end_sector']))


def toc_from_file(path):
    """Reads EAC / XLD / fre:ac log files, generates MusicBrainz disc TOC listing for use as discid.

    Warning: may work wrong for discs having data tracks. May generate wrong
    results on other non-standard cases."""
    encoding = detect_file_encoding(path)
    with open(path, 'r', encoding=encoding) as f:
        return calculate_mb_toc_numbers(filter_toc_entries(f))

# Edit by dm4

import sys
import discid

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <logfile>")
        sys.exit(1)
    toc = list(toc_from_file(sys.argv[1]))
    disc = discid.put(toc[0], toc[1], toc[2], toc[3:])
    print(f"Disc ID:\n{disc.id}\n")
    print(f"TOC:\n{disc.toc_string}\n")
    print(f"Submit to MusicBrainz:\n{disc.submission_url}\n")

if __name__ == "__main__":
    main()