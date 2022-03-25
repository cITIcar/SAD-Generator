"""
Augmentation of real world images with start lines.

This GUI allows to transform real world camera images in bird's-eye-view
and add a start line inside it. After the adding of the startline, the image
will be transformed back into camera perspective.
"""

import os
import sys
import numpy as np
import cv2
import json
import glob
from numpy.linalg import inv
from manual_augment import ManualAugment


class Startline(ManualAugment):
    """
    Augmentation of real world images with start line images.

    Attributes
    ----------
    start_line_rows : int
        Number of rows that the startline has.
    start_line_colums : int
        Number of colums that the startline has.
    patch_size : int
        Amount of pixels that sedcribe the height and width of a
        single square.
    """

    def __init__(self, config_path="config1.json"):
        """Define class attributes."""
        super().__init__(config_path)
        with open('config1.json', 'r') as f:
            config = json.load(f)

        startline_config = config["augmentation_config"]
        self.start_line_rows = startline_config["start_line_rows"]
        self.start_line_colums = startline_config["start_line_colums"]
        self.patch_size = startline_config["patch_size"]
        self.output_size = config["output_size"]

    def draw_startline(self):
        """
        Draw the basic shape of the startline.

        Parameters
        ----------
        None.

        Returns
        -------
        start_line_image : int array
            Synthetic image that represents a real world start line
        start_line_label : int array
            label of the start line
        """
        white_patch = np.random.randint(150, 255,
                                        (self.patch_size, self.patch_size))
        start_line_image = np.random.randint(0, 10, (
                self.patch_size*self.start_line_rows,
                self.patch_size*self.start_line_colums))
        start_line_label = np.ones((
                self.patch_size*self.start_line_rows,
                self.patch_size *
                self.start_line_colums))*self.overlay_label_value

        for i in range(self.start_line_rows):
            y_start = i * self.patch_size
            y_end = (i + 1)*self.patch_size

            for j in range(self.start_line_colums):
                if 2*j+1+i % 2 > self.start_line_colums:
                    break
                x_start = (2*j+i % 2)*self.patch_size
                x_end = (2*j+1+i % 2)*self.patch_size
                start_line_image[y_start:y_end, x_start:x_end] = white_patch

        return start_line_image, start_line_label

    def get_birds_eye_view(self, camera_image, camera_label):
        """
        Transform the perspective of the camera images into bird's-eye-view.

        Parameters
        ----------
        camera_image : numpy array
            Image in camera perspective
        camera_label : numpy array
            label in camera perspective

        Returns
        -------
        bird_image : numpy array
            Image in bird's-eye-view
        bird_label : numpy array
            label in bird's-eye-view
        """
        bird_label = cv2.warpPerspective(
                camera_label, inv(self.renderer.h_label),
                (3000, 3000), flags=self.interpolation)
        bird_image = cv2.warpPerspective(
                camera_image, inv(self.renderer.h_label),
                (3000, 3000), flags=self.interpolation)
        return bird_image, bird_label

    def get_camera_view(self, bird_image, bird_label):
        """
        Transform the perspective of bird's-eye-view images into camera view.

        Parameters
        ----------
        bird_image : numpy array
            Image in bird's-eye-view
        bird_label : numpy array
            label in bird's-eye-view

        Returns
        -------
        camera_image : numpy array
            Image in camera perspective
        camera_label : numpy array
            label in camera perspective
        """
        camera_label = cv2.warpPerspective(
                bird_label, self.renderer.h_label, self.output_size,
                flags=self.interpolation)
        camera_image = cv2.warpPerspective(
                bird_image, self.renderer.h_label, self.output_size,
                flags=self.interpolation)
        return camera_image, camera_label

    def merge_bird_overlay(self, overlay_img, overlay_label, bird_img,
                           bird_label, camera_img, camera_label):
        """
        Add together background and overlay for the image and label.

        Parameters
        ----------
        overlay_img : numpy array
            Image of overlay in bird's-eye-view
        overlay_label : numpy array
            label of overlay in bird's-eye-view
        bird_img : numpy array
            Image of overlay in bird's-eye-view
        bird_label : numpy array
            label of background in bird's-eye-view

        Returns
        -------
        bird_img : numpy array
            Image of background with merged overlay in bird's-eye-view
        bird_label : numpy array
            label of background with merged overlay in bird's-eye-view
        camera_img : numpy array
            Image of background with merged overlay in camera view
        camera_label : numpy array
            label of background with merged overlay in camera view
        """
        # Delete the startline where there is no road below it
        overlay_img[np.logical_or(bird_label >= 175, bird_label <= 25)] = 0
        # Only annotate startline where there was road below it
        overlay_label[np.logical_or(bird_label >= 175, bird_label <= 25)] = 0
        bird_label[np.logical_and(
                np.logical_and(bird_label <= 175, overlay_label >= 225),
                bird_label >= 25)] = 250
        bird_img = np.clip(bird_img + overlay_img, 0, 255)
        overlay_camera_img, overlay_camera_label = self.get_camera_view(
                overlay_img, overlay_label)

        camera_label = np.clip(overlay_camera_label + camera_label, 0, 255)
        camera_img = np.clip(overlay_camera_img + camera_img, 0, 255)

        return (bird_img, bird_label, camera_img, camera_label)

    def visualize_augmentation(self, bird_label, bird_img, camera_img,
                               camera_label, index):
        """
        Visualize the label and image of the bird's-eye-view and camera view.

        Parameters
        ----------
        bird_img : numpy array
            Image of background with merged overlay in bird's-eye-view
        bird_label : numpy array
            label of background with merged overlay in bird's-eye-view
        camera_img : numpy array
            Image of background with merged overlay in camera view
        camera_label : numpy array
            label of background with merged overlay in camera view
        index : int
            unique identifier of one augmented annotated sample

        Returns
        -------
        key : int
            Value of pressed key
        """
        bird_img_resized = cv2.resize(bird_img, (1000, 1000),
                                      interpolation=self.interpolation)
        bird_label_resized = cv2.resize(bird_label, (1000, 1000),
                                        interpolation=self.interpolation)

        bird_label_resized = bird_label_resized[0:500, :]
        bird_img_resized = bird_img_resized[0:500, :]

        camera_label_resized = cv2.resize(camera_img, (1000, 1000),
                                          interpolation=self.interpolation)
        camera_img_resized = cv2.resize(camera_label, (1000, 1000),
                                        interpolation=self.interpolation)

        label = np.concatenate([bird_label_resized,
                                camera_label_resized], axis=0)
        image = np.concatenate([bird_img_resized, camera_img_resized], axis=0)
        result = np.concatenate([label, image], axis=1)

        result = cv2.resize(result, (667, 500))
        cv2.imshow("result", result.astype(np.uint8))
        key = cv2.waitKey(0)
        if key == ord(" "):
            cv2.imwrite(self.label_write_path + str(index) +
                        ".png", camera_label)
            cv2.imwrite(self.img_write_path + str(index) +
                        ".png", camera_img)

        return key


if __name__ == "__main__":
    if len(sys.argv) > 1:
        startline = Startline(sys.argv[1])
    else:
        startline = Startline()

    index = 0

    startline_img, startline_label = startline.draw_startline()

    labels_list = glob.glob(startline.label_input_path + "/*.png") if os.path.isdir(startline.label_input_path) else glob.glob(startline.label_input_path)
    images_list = glob.glob(startline.img_input_path + "/*.png") if os.path.isdir(startline.img_input_path) else glob.glob(startline.img_input_path)

    startline_img_, startline_label_ = startline.create_overlay(startline_img)

    if len(labels_list) == 0:
        print(f"no annotated images found under {}".format(self.label_input_path))

    if len(images_list) == 0:
        print(f"no images found under {self.img_input_path}")

    for label_path, img_path in zip(labels_list, images_list):
        camera_img, camera_label = startline.import_annotated_data(
                img_path, label_path)
        bird_img, bird_label = startline.get_birds_eye_view(
                camera_img, camera_label)

        startline_img, startline_label = (np.copy(startline_img_),
                                         np.copy(startline_label_))
        key = 0
        while key != ord(" ") or ord("q"):

            startline_img, startline_label = startline.transform_image(
                    startline_img, startline_label, key)

            (bird_img_n, bird_label_n,
             camera_img_n, camera_label_n) = startline.merge_bird_overlay(
                    np.copy(startline_img), np.copy(startline_label),
                    np.copy(bird_img), np.copy(bird_label),
                    np.copy(camera_img), np.copy(camera_label))
            key = startline.visualize_augmentation(bird_label_n, bird_img_n,
                                                   camera_img_n, camera_label_n,
                                                   index)

            if key == ord(" "):
                break
            if key == ord("q"):
                break
            if key == ord("x"):
                exit()
        index += 1
