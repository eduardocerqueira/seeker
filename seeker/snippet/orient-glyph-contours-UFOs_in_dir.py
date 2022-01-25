#date: 2022-01-25T16:50:56Z
#url: https://api.github.com/gists/98e98405d1575672f86fdce307c11b08
#owner: https://api.github.com/users/arrowtype

"""
    Go through UFOs in directory and orient contour paths in a counter-clockwise direction,
    as they should be for cubic/postscript/CFF/"OTF" curves.

    DEPENDENCIES:
    - fontParts
    - ufonormalizer

    USAGE:
    On the command line, call this script and add a directory of UFOs as an argument. 
    Add "--normalize" to normalize the UFOs after saving.

        python3 <path_to_script>/orient-glyph-contours-UFOs_in_dir.py <path_to_directory_of_UFOs> --normalize

    DISCLAIMERS: 
    - May break compatibility on some glyphs, as start points can change.
    - May result in some broken drawings for certain edge cases (e.g. if a counter path 
      somehow goes further left than the main exterior path).
    - Doesn't handle glyphs with more than two counters, unless they all have the same 
      path direction (i.e. no counters), but adds these to a list for manual review.
    - May not work for your project. ¯\_(ツ)_/¯ ALWAYS USE GIT / VERSION CONTROL or another 
      form of backup before running any script like this!

    LICENSE:
    - MIT. Use/remix this if you want to.

"""

import argparse
import os
from fontParts.world import *
from ufonormalizer import normalizeUFO


def orientPathsCCW(glyph):
    """
        Orient all paths in glyph counter-clockwise.
        Only use this on glyphs with no counter/cutout shapes.
    """
    for contour in glyph:
        if contour.clockwise == True:
            contour.reverse()


def main():

    # get arguments from argparse
    args = parser.parse_args()

    sourceFolderPath = args.dir

    # get UFO paths and open each of them
    for subPath in os.listdir(sourceFolderPath):
        if subPath.endswith(".ufo"):
            ufoPath = os.path.join(sourceFolderPath, subPath)

            f = OpenFont(ufoPath, showInterface=False)

            print(f"Analyzing: {f.info.styleName}...")

            # a list to track glyphs with many contours, for manual review
            manyContours = []

            # go through glyphs in the font
            for g in f:

                # if all contours have the same path direction, orient the paths CCW
                if len(set([c.clockwise for c in g.contours])) == 1:
                   orientPathsCCW(g)

                # else if contours have more than one direction
                else:
                    # check for exactly two contours
                    if len(g.contours) == 2:

                        # find outer shape, then make it CCW
                        # if first contour is further left than second contour, assume it is exterior and orient first to CCW
                        # bounds is (xMin, yMin, xMax, yMax)
                        if g.contours[0].bounds[0] < g.contours[1].bounds[0]:
                            if g.contours[0].clockwise:
                                g.contours[0].reverse()
                                g.contours[1].reverse()

                        # if first contour is not further left, assume it is the counter, and make it CW
                        else:
                            if g.contours[0].clockwise == False:
                                g.contours[0].reverse()
                                g.contours[1].reverse()

                    # if too many contours, just add to a list for manual review/handling
                    if len(g.contours) == 3:
                        manyContours.append(g.name)

            # report list for manual review
            if len(manyContours) >= 1:
                print("The following glyphs have more than two contours, and multiple path directions:")
                print(" ".join(manyContours))
                print()

            f.save()

            if args.normalize:
                normalizeUFO(ufoPath)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Orient path directions in all glyphs for all UFOs in a directory.')

    parser.add_argument('dir',
                        help='Relative path to a directory of one or more UFO font sources')
    parser.add_argument("-n", "--normalize",
                        action='store_true',
                        help='Normalizes UFOs with ufoNormalizer.')
    main()
