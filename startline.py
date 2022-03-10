"""
Augmentation of real world images with start lines.
This GUI allows to transform real world camera images in bird's-eye-view
and add a start line inside it. After the adding of the startline, the image
will be transformed back into camera perspective.
"""

import numpy as np
import cv2
import json
import glob
from numpy.linalg import inv
from manual_augment import ManualAugment


class Startline(ManualAugment):
    """
    This class is made for the augmentation of real world images with start
    line images.

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

    def __init__(self):
        """
        Define class attributes.
        """
        super().__init__()
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
        start_line_mask : int array
            Annotation of the start line
        """

        white_patch = np.random.randint(150, 255,
                                        (self.patch_size, self.patch_size))
        start_line_image = np.random.randint(0, 10, (
                self.patch_size*self.start_line_rows,
                self.patch_size*self.start_line_colums))
        start_line_mask = np.ones((
                self.patch_size*self.start_line_rows,
                self.patch_size *
                self.start_line_colums))*self.overlay_mask_value

        for i in range(self.start_line_rows):
            y_start = i * self.patch_size
            y_end = (i + 1)*self.patch_size

            for j in range(self.start_line_colums):
                if 2*j+1+i % 2 > self.start_line_colums:
                    break
                x_start = (2*j+i % 2)*self.patch_size
                x_end = (2*j+1+i % 2)*self.patch_size
                start_line_image[y_start:y_end, x_start:x_end] = white_patch

        return start_line_image, start_line_mask

    def get_birds_eye_view(self, camera_image, camera_mask):
        """
        Transform the perspective of the camera images into bird's-eye-view.

        Parameters
        ----------
        camera_image : numpy array
            Image in camera perspective
        camera_mask : numpy array
            Annotation in camera perspective

        Returns
        -------
        bird_image : numpy array
            Image in bird's-eye-view
        bird_mask : numpy array
            Annotation in bird's-eye-view
        """
        bird_mask = cv2.warpPerspective(
                camera_mask, inv(self.renderer.h_segmentation),
                (3000, 3000), flags=self.interpolation)
        bird_image = cv2.warpPerspective(
                camera_image, inv(self.renderer.h_segmentation),
                (3000, 3000), flags=self.interpolation)
        return bird_image, bird_mask

    def get_camera_view(self, bird_image, bird_mask):
        """
        Transform the perspective of bird's-eye-view images into camera view.

        Parameters
        ----------
        bird_image : numpy array
            Image in bird's-eye-view
        bird_mask : numpy array
            Annotation in bird's-eye-view

        Returns
        -------
        camera_image : numpy array
            Image in camera perspective
        camera_mask : numpy array
            Annotation in camera perspective
        """
        camera_mask = cv2.warpPerspective(
                bird_mask, self.renderer.h_segmentation, self.output_size,
                flags=self.interpolation)
        camera_image = cv2.warpPerspective(
                bird_image, self.renderer.h_segmentation, self.output_size,
                flags=self.interpolation)
        return camera_image, camera_mask

    def merge_bird_overlay(self, overlay_img, overlay_mask, bird_img,
                           bird_mask, camera_img, camera_mask):
        """
        Add together background and overlay for the image and annotation.

        Parameters
        ----------
        overlay_img : numpy array
            Image of overlay in bird's-eye-view
        overlay_mask : numpy array
            Annotation of overlay in bird's-eye-view
        bird_img : numpy array
            Image of overlay in bird's-eye-view
        bird_mask : numpy array
            Annotation of background in bird's-eye-view

        Returns
        -------
        bird_img : numpy array
            Image of background with merged overlay in bird's-eye-view
        bird_mask : numpy array
            Annotation of background with merged overlay in bird's-eye-view
        camera_img : numpy array
            Image of background with merged overlay in camera view
        camera_mask : numpy array
            Annotation of background with merged overlay in camera view
        """
        # Delete the startline where there is no road below it
        overlay_img[np.logical_or(bird_mask >= 175, bird_mask <= 25)] = 0
        # Only annotate startline where there was road below it
        overlay_mask[np.logical_or(bird_mask >= 175, bird_mask <= 25)] = 0
        bird_mask[np.logical_and(
                np.logical_and(bird_mask <= 175, overlay_mask >= 225),
                bird_mask >= 25)] = 250
        bird_img = np.clip(bird_img + overlay_img, 0, 255)
        overlay_camera_img, overlay_camera_mask = self.get_camera_view(
                overlay_img, overlay_mask)

        camera_mask = np.clip(overlay_camera_mask + camera_mask, 0, 255)
        camera_img = np.clip(overlay_camera_img + camera_img, 0, 255)

        return (bird_img, bird_mask, camera_img, camera_mask)

    def visualize_augmentation(self, bird_mask, bird_img, camera_img,
                               camera_mask, index):
        """
        Visualize the mask and image of the bird's-eye-view and camera view.

        Parameters
        ----------
        bird_img : numpy array
            Image of background with merged overlay in bird's-eye-view
        bird_mask : numpy array
            Annotation of background with merged overlay in bird's-eye-view
        camera_img : numpy array
            Image of background with merged overlay in camera view
        camera_mask : numpy array
            Annotation of background with merged overlay in camera view
        index : int
            unique identifier of one augmented annotated sample

        Returns
        -------
        key : int
            Value of pressed key
        """
        bird_img_resized = cv2.resize(bird_img, (1000, 1000),
                                      interpolation=self.interpolation)
        bird_mask_resized = cv2.resize(bird_mask, (1000, 1000),
                                       interpolation=self.interpolation)

        bird_mask_resized = bird_mask_resized[0:500, :]
        bird_img_resized = bird_img_resized[0:500, :]

        camera_mask_resized = cv2.resize(camera_img, (1000, 1000),
                                         interpolation=self.interpolation)
        camera_img_resized = cv2.resize(camera_mask, (1000, 1000),
                                        interpolation=self.interpolation)

        mask = np.concatenate([bird_mask_resized, camera_mask_resized], axis=0)
        image = np.concatenate([bird_img_resized, camera_img_resized], axis=0)
        result = np.concatenate([mask, image], axis=1)

        result = cv2.resize(result, (667, 500))
        cv2.imshow("result", result.astype(np.uint8))
        key = cv2.waitKey(0)
        if key == ord(" "):
            cv2.imwrite(self.mask_write_path + str(index) +
                        ".png", camera_mask)
            cv2.imwrite(self.img_write_path + str(index) +
                        ".png", camera_img)

        return key


if __name__ == "__main__":
    startline = Startline()
    index = 0

    startline_img, startline_mask = startline.draw_startline()

    annotations_list = glob.glob("./real_dataset/annotations/set_*/*.png")

    startline_img_, startline_mask_ = startline.create_overlay(startline_img)

    if len(annotations_list) == 0:
        print("no annotated images found under the path")

    for mask_path in annotations_list:
        img_path = mask_path.replace("annotations", "images")
        camera_img, camera_mask = startline.import_annotated_data(
                img_path, mask_path)
        bird_img, bird_mask = startline.get_birds_eye_view(
                camera_img, camera_mask)

        startline_img, startline_mask = (np.copy(startline_img_),
                                         np.copy(startline_mask_))
        key = 0
        while key != ord(" ") or ord("q"):

            startline_img, startline_mask = startline.transform_image(
                    startline_img, startline_mask, key)

            (bird_img_n, bird_mask_n,
             camera_img_n, camera_mask_n) = startline.merge_bird_overlay(
                    np.copy(startline_img), np.copy(startline_mask),
                    np.copy(bird_img), np.copy(bird_mask),
                    np.copy(camera_img), np.copy(camera_mask))
            key = startline.visualize_augmentation(bird_mask_n, bird_img_n,
                                                   camera_img_n, camera_mask_n,
                                                   index)

            if key == ord(" "):
                break
            if key == ord("q"):
                break
            if key == 27:
                exit()
        index += 1

