#!/usr/bin/env python3

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
    road_generator = road.Road(config)
    renderer = render.Renderer(config)

    # initialize disturbances from the config file
    objects = [
        globals()[obj](*params, config=config)
        for obj, params in config["disturbances"].items() ]

    images_base_path = config["paths"]["images_output_path"].format(splitname=splitname)
    annotations_base_path = config["paths"]["annotations_output_path"].format(splitname=splitname)
    image_pattern = config["paths"]["output_file_pattern"]

    idx = 0
    running = True
    while running:
        t1 = time.time()
        image, image_segment, drive_points, _, camera_angles = road_generator.build_road()
        renderer.update_ground_plane(image, image_segment, objects)

        for point, angle in zip(drive_points, camera_angles):
            renderer.update_position(point, angle)
            perspective_nice, perspective_segment = renderer.render_images()
            perspective_nice = np.clip(perspective_nice, 0, 255).astype(np.uint8)
            perspective_segment = perspective_segment.astype(np.uint8)

            if config["debug"]:
                cv2.imshow(f"nice {splitname}", perspective_nice)
                cv2.imshow(f"segment {splitname}", perspective_segment)
                cv2.waitKey(0)
            else:
                cv2.imwrite(images_base_path + "/" + image_pattern.format(idx=output_idcs[idx]), perspective_nice)
                cv2.imwrite(annotations_base_path + "/" + image_pattern.format(idx=output_idcs[idx]), perspective_segment)

            if idx >= len(output_idcs) - 1:
                running = False
                break
            idx += 1

        if running:
            print(f"\033[1A\033[K{len(drive_points) / (time.time() - t1):.5} fps")


def generate_augmented(config, splitname, output_idcs):
    annotations_input_path = config["paths"]["manual_annotations_input_path"]
    images_input_path = config["paths"]["manual_images_input_path"]

    images_base_path = config["paths"]["images_output_path"].format(splitname=splitname)
    annotations_base_path = config["paths"]["annotations_output_path"].format(splitname=splitname)
    image_pattern = config["paths"]["output_file_pattern"]

    augment.augment_dataset(
        annotations_input_path, images_input_path, 
        annotations_base_path, images_base_path, output_idcs, config)


def init_paths(config):
    output_path_annotations = config["paths"]["annotations_output_path"]
    output_path_images = config["paths"]["images_output_path"]

    os.makedirs(output_path_annotations.format(splitname="train_split"), exist_ok=True)
    os.makedirs(output_path_annotations.format(splitname="validation_split"), exist_ok=True)
    os.makedirs(output_path_annotations.format(splitname="test_split"), exist_ok=True)

    os.makedirs(output_path_images.format(splitname="train_split"), exist_ok=True)
    os.makedirs(output_path_images.format(splitname="validation_split"), exist_ok=True)
    os.makedirs(output_path_images.format(splitname="test_split"), exist_ok=True)


if __name__ == "__main__":
    args = parser.parse_args()

    config = Config(args.config, debug=args.debug)
    if config["seed"]:
        random.seed(config["seed"])
        np.random.seed(config["seed"])

    init_paths(config)

    idcs_train = list(range(config["splits"]["train_split"]["size"]))
    idcs_validation = list(range(config["splits"]["validation_split"]["size"]))
    idcs_test = list(range(config["splits"]["test_split"]["size"]))

    if config["shuffle"]:
        random.shuffle(idcs_train)
        random.shuffle(idcs_validation)
        random.shuffle(idcs_test)

    print("generating synthetic:")
    print("train split\n")
    generate_synthetic(
        config,
        "train_split",
        idcs_train[round(
            (1 - config["splits"]["train_split"]["fraction_synthetic"]) *
            config["splits"]["train_split"]["size"]):])
    print("validation split\n")
    generate_synthetic(
        config,
        "validation_split",
        idcs_validation[round(
            (1 - config["splits"]["validation_split"]["fraction_synthetic"]) *
            config["splits"]["validation_split"]["size"]):])
    print("test split\n")
    generate_synthetic(
        config,
        "test_split",
        idcs_test[round(
            (1 - config["splits"]["test_split"]["fraction_synthetic"]) *
            config["splits"]["test_split"]["size"]):])

    print("generating augmented:")
    print("train split")
    generate_augmented(
        config,
        "train_split",
        idcs_train[:round(
            config["splits"]["train_split"]["fraction_augmented"] *
            config["splits"]["train_split"]["size"])])
    print("validation split")
    generate_augmented(
        config,
        "validation_split",
        idcs_validation[:round(
            config["splits"]["validation_split"]["fraction_augmented"] *
            config["splits"]["validation_split"]["size"])])
    print("test split")
    generate_augmented(
        config,
        "test_split",
        idcs_test[:round(
            config["splits"]["test_split"]["fraction_augmented"] *
            config["splits"]["test_split"]["size"])])


