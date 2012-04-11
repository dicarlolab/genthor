#!/usr/bin/env python

import os
import subprocess
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

