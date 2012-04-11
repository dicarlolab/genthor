#!/usr/bin/env python

import os
import sys


def convert(in_img, out_img, tonemap_alg=None, gamma=2.0):
    """ Converts an hdr img to a jpg."""

    #in_img = "INTERIOR_44ST.hdr"
    #out_img = "out.jpg"

    pfstmo = (
        "drago03",
        "durand02",
        "fattal02",
        "mantiuk06",
        "mantiuk08",
        "pattanaik00",
        "reinhard02",
        "reinhard05",
        )

    # Default tonemap algorithm
    if tonemap_alg is None:
        tonemap_alg = pfstmo[1]

    # Command
    cmdstr = ("pfsin %s | pfstmo_%s | pfsgamma -g %.1f | pfsout %s"
              % (in_img, tonemap_alg, gamma, out_img))

    # Run it
    os.system(cmdstr)


def run(path):
    print path

    hdrfiles = os.listdir(path)

    for hdrfile in hdrfiles:
        in_img = os.path.join(path, hdrfile)
        out_img = os.path.splitext(in_img)[0] + ".jpg"
        print in_img, os.path.isfile(in_img)
        print out_img, os.path.isfile(out_img)
        print
        convert(in_img, out_img)
    

if __name__ == "__main__":

    run(sys.argv[1])
