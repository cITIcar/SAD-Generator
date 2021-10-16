#!/usr/bin/env python3

import argparse
import json
import glob
import numpy as np
import cv2
import os

CLASS_SEPERATION = 50
MAX_CLASS = 200
TOLERANCE = 40

parser = argparse.ArgumentParser(description="swap classes for annotated images")
parser.add_argument(
    "input_path", metavar="input-path", type=str,
    help="path to the input files")
parser.add_argument(
    "output_path", metavar="output-path", type=str,
    help="path to the output files")
parser.add_argument(
    "value_map", metavar="value-mappings", type=str,
    help="a python-dict string mapping class values to another class")

if __name__ == "__main__":
    args = parser.parse_args()
    os.makedirs(args.output_path, exist_ok=True)

    value_map = eval(args.value_map)

    for file in glob.glob(args.input_path + "/*"):
        image = cv2.imread(file)

        if image is None:
            continue

        image = image.clip(0, MAX_CLASS)
        image = image / CLASS_SEPERATION
        image = image.round().astype(np.uint8) * CLASS_SEPERATION

        new = image.copy()

        for key, value in value_map.items():
            new[abs(image - key) < TOLERANCE] = value
        cv2.imshow("new", new)
        cv2.waitKey(1)

        cv2.imwrite(args.output_path + "/" + os.path.basename(file).replace("jpg", "png"), new)
