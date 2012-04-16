#!/usr/bin/env python

import os
from subprocess import Popen
from subprocess import PIPE
import sys


def convert(in_img, out_img):
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

    tonemap_alg = pfstmo[0]
    bias = 0.6
    gamma = 0.7

    # Commands
    p1 = Popen(["pfsin", in_img], bufsize=-1, stdout=PIPE)
    p2 = Popen(["pfstmo_%s" % tonemap_alg, "-b", str(bias)],
               bufsize=-1, stdin=p1.stdout, stdout=PIPE)
    p3 = Popen(["pfsgamma", "-g", str(gamma)],
               bufsize=-1, stdin=p2.stdout, stdout=PIPE)
    p4 = Popen(["pfsout", out_img], bufsize=-1, stdin=p3.stdout)
    p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
    p2.stdout.close()  # Allow p2 to receive a SIGPIPE if p3 exits.
    p3.stdout.close()  # Allow p3 to receive a SIGPIPE if p4 exits.
    output = p4.communicate()


def run(path):
    print path

    hdrfiles = [fn for fn in os.listdir(path)
                if os.path.splitext(fn)[1] == ".hdr"]

    for hdrfile in hdrfiles:
        in_img = os.path.join(path, hdrfile)
        out_img = os.path.splitext(in_img)[0] + ".jpg"
        print in_img, os.path.isfile(in_img)
        print out_img, os.path.isfile(out_img)
        print
        convert(in_img, out_img)
    

if __name__ == "__main__":

    run(sys.argv[1])
