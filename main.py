# !/usr/bin/env python3
"""Commands for creating augmented and synthetic data."""

import os
import time
import random
import argparse
import numpy as np
import cv2

from config import Config
import road
import render
from disturbances import *
import augment


parser = argparse.ArgumentParser(description="generate training data")
parser.add_argument(
    "--config", metavar="config file", type=str, required=True,
    help="path to the config file")
parser.add_argument(
    "--debug", action="store_true",
    help="display the video stream instead of saving the images")

def generate_synthetic(config, splitname, output_idcs):
    """Create the synthetic fraction of the split defined in config.

    Parameters
    ----------
    config : Config
        The configuration.
    splitname : str
        The name of the split.
    output_idcs : List[int]
        The list of indices for which images need to be created

    Returns
    -------
    None.

    """
    road_generator = road.Road(config)
    renderer = render.Renderer(config)

    # instantiate disturbances
    objects = []
    for d in config["disturbances"]:
        (obj, params), = list(d.items())
        objects.append(globals()[obj](*params, config=config))

    images_base_path = config["paths"]["images_output_path"].format(
        splitname=splitname)
    annotations_base_path = config["paths"]["annotations_output_path"].format(
        splitname=splitname)
    image_pattern = config["paths"]["output_file_pattern"]

    idx = 0
    running = True
    while running:
        t1 = time.time()
        (image, image_segment,
         drive_points, _, camera_angles) = road_generator.build_road()
        renderer.update_ground_plane(image, image_segment, objects)

        for p_idx, (point, angle) in enumerate(zip(drive_points,
                                                   camera_angles)):
            renderer.update_position(point, angle)
            perspective_nice, perspective_segment = renderer.render_images()
            perspective_nice = np.clip(perspective_nice,
                                       0, 255).astype(np.uint8)
            perspective_segment = perspective_segment.astype(np.uint8)

            if config["debug"]:
                cv2.imshow(f"nice {splitname}", perspective_nice)
                cv2.imshow(f"segment {splitname}", perspective_segment)
                cv2.waitKey(1)
            else:
                cv2.imwrite(images_base_path +
                            "/" + image_pattern.format(idx=output_idcs[idx]),
                            perspective_nice)
                cv2.imwrite(annotations_base_path +
                            "/" + image_pattern.format(idx=output_idcs[idx]),
                            perspective_segment)

            if idx >= len(output_idcs) - 1:
                running = False
                break
            idx += 1

        print(f"\033[1A\033[K{p_idx / (time.time() - t1):.5}" +
              " fps, {idx + 1}/{len(output_idcs)}")

def generate_augmented(config, splitname, output_idcs):
    """Create the augmented fraction of the split defined in config.

    Parameters
    ----------
    config : Config
        The configuration.
    splitname : str
        The name of the split.
    output_idcs : List[int]
        The list of indices for which images need to be created

    Returns
    -------
    None.

    """
    annotations_input_path = config["paths"]["manual_annotations_input_path"]
    images_input_path = config["paths"]["manual_images_input_path"]

    images_base_path = config["paths"]["images_output_path"].format(
        splitname=splitname)
    annotations_base_path = config["paths"]["annotations_output_path"].format(
        splitname=splitname)
    image_pattern = config["paths"]["output_file_pattern"]

    augment.augment_dataset(
        annotations_input_path, images_input_path,
        annotations_base_path, images_base_path, output_idcs, config)


if __name__ == "__main__":
    args = parser.parse_args()

    config = Config(args.config, debug=args.debug)
    if config["seed"]:
        random.seed(config["seed"])
        np.random.seed(config["seed"])

    output_path_annotations = config["paths"]["annotations_output_path"]
    output_path_images = config["paths"]["images_output_path"]

    for name, split in config["splits"].items():
        os.makedirs(output_path_annotations.format(splitname=name),
                    exist_ok=True)
        os.makedirs(output_path_images.format(splitname=name), exist_ok=True)
        print("generating split", name)
        print("synthetic")
        print()
        idcs = list(range(split["size"]))
        if config["shuffle"]:
            random.shuffle(idcs)
        generate_synthetic(
            config,
            name,
            idcs[round(
                (1 - split["fraction_synthetic"]) *
                split["size"]):])

        print("augmented")
        generate_augmented(
            config,
            name,
            idcs[:round(
                split["fraction_augmented"] *
                split["size"])])
