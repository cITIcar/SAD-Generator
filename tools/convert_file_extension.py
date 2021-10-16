#!/usr/bin/env python3

import argparse
import glob
import os
import cv2

parser = argparse.ArgumentParser(description="convert images from one extension to another")
parser.add_argument(
    "input_path", metavar="input-path", type=str,
    help="path to the input files")
parser.add_argument(
    "output_path", metavar="output-path", type=str,
    help="path to the output files")
parser.add_argument(
    "from_ext", metavar="from-ext", type=str,
    help="path to the input files")
parser.add_argument(
    "to_ext", metavar="to-ext", type=str,
    help="path to the output files")

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    for file in glob.glob(args.input_path + "/*"):
        image = cv2.imread(file)

        if image is None:
            continue

        cv2.imwrite(args.output_path + "/" + os.path.basename(file).replace(args.from_ext, args.to_ext), image)
