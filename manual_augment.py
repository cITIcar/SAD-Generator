"""
Augmentation of real world images with overlays.

Via key presses the user can rotate and translate the overlay inside the
real world background.
Following keys exist:
    "w"/"s": Translation along y-axis
    "a"/"d": Translation along x-axis
    "e"/"r": Rotation of overlay
    Esc: Abort augmentation
    Space: Save augmented sample
"""

import numpy as np
import cv2
import json
import math
import config
import render
import imutils
from numpy.linalg import inv


class ManualAugment:
    """
    Template for the manual augmentation of real world images.

    This GUI allows to transform real world camera
    images in bird's-eye-view and add a overlay inside it. After the adding
    of the overlay, the image will be transformed back into camera perspective.

    Attributes
    ----------
    translation_step : int
        How many pixels the startline moves every time the key is pressed
    rotation_step : int
        How many degrees the startline rotates every time the key is pressed
    angle : int
        Initial angle of startline
    offset_x : int
        Initial offset of startline along x axis
    offset_y : int
        Initial offset of startline along y axis
    mask_write_path : String
        Path where augmented mask is written to
    img_write_path : String
        Path where augmented image is written to
    renderer : render Object
        Object of Render Class. Defined in config.py.
    interpolation : OpenCV parameter
        How pixels will be interpolated in resize and warping operations
    """

    def __init__(self, config_path):
        """Define class attributes."""
        f = open(config_path, "r")
        render_config = config.Config(config_path, debug=False)
        aug_config = json.load(f)["augmentation_config"]
        self.translation_step = aug_config["translation_step"]
        self.rotation_step = aug_config["rotation_step"]
        self.angle = aug_config["angle"]
        self.offset_x = aug_config["offset_x"]
        self.offset_y = aug_config["offset_y"]
        self.mask_write_path = aug_config["mask_write_path"]
        self.img_write_path = aug_config["img_write_path"]
        self.mask_input_path = aug_config["mask_input_path"]
        self.img_input_path = aug_config["img_input_path"]
        self.img_overlay_path = aug_config["img_object_path"]
        self.mask_overlay_path = aug_config["mask_object_path"]
        self.camera_x_size = render_config["output_size"][0]
        self.camera_y_size = render_config["output_size"][1]
        self.overlay_mask_value = aug_config["overlay_mask_value"]
        self.annotation_buffer = aug_config["annotation_buffer"]

        # Define render object for perspective transform
        camera_yaw = math.pi
        position_camera = (aug_config["position_camera_x"],
                           aug_config["position_camera_y"])
        self.renderer = render.Renderer(render_config)
        self.renderer.update_position(position_camera, camera_yaw)
        self.interpolation = cv2.INTER_NEAREST

    def import_annotated_data(self, img_path, mask_path):
        """
        Load array from image and mask of annotated sample.

        Parameters
        ----------
        img_path : String
            Path to image of annotated sample
        mask_path : String
            Path to mask of annotated smaple

        Returns
        -------
        img : numpy array
            Image of annotated sample
        mask : numpy array
            Mask of annotated sample
        """
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            print("Image or Annotation not found.")
        if img.shape != mask.shape:
            print("Shape of Image and Annotation is not consistent.")
        if img.shape != (self.camera_y_size, self.camera_x_size):
            print("Image size is not consistent with configuration.")
        return img, mask

    def load_overlay(self):
        """
        Load a annotated sample of an overlaying object.

        The overlay has to be in bird's-eye-view.

        Parameters
        ----------
        None.

        Returns
        -------
        overlay_img : int array
            Synthetic image that represents a real world object
        overlay_mask : int array
            Annotation of the object
        """
        overlay_img = cv2.imread(self.img_overlay_path,
                                 cv2.IMREAD_GRAYSCALE)
        overlay_mask = cv2.imread(self.mask_overlay_path,
                                  cv2.IMREAD_GRAYSCALE)
        return overlay_img, overlay_mask

    def create_overlay(self, overlay_img):
        """
        Insert the overlay into a black background.

        The overlay has to be in bird's-eye-view.

        Parameters
        ----------
        overlay_img : numpy array
            image of the overlay

        Returns
        -------
        img : int array
            Overlay image inside a black background
        mask : int array
            Annotation of the object inside a black background
        """
        img = np.zeros((3000, 3000))
        mask = np.zeros((3000, 3000))

        x_size = overlay_img.shape[0]
        y_size = overlay_img.shape[1]

        img[self.offset_x:self.offset_x + x_size,
            self.offset_y:self.offset_y + y_size] = overlay_img
        mask[self.offset_x:self.offset_x + x_size,
             self.offset_y:self.offset_y + y_size] = self.overlay_mask_value

        return img, mask

    def transform_image(self, image, mask, key):
        """
        Translate or rotate image and annotation.

        Depending of the key pressed by the user the image and annotation is
        translated or rotated with the value defined in the config file.

        Parameters
        ----------
        image : numpy array
            Image that contains overlay.
        mask : numpy array
            Annotation of the image with the overlay.
        key : int
            Value of pressed key.

        Returns
        -------
        image : numpy array
            Altered image that contains the overlay.
        mask : numpy array
            Altered annotation that contains the overlay.
        """
        if(key == ord("a")):
            image = imutils.translate(image, self.translation_step, 0)
            mask = imutils.translate(mask, self.translation_step, 0)

        elif(key == ord("d")):
            image = imutils.translate(image, -self.translation_step, 0)
            mask = imutils.translate(mask, -self.translation_step, 0)

        if(key == ord("w")):
            image = imutils.translate(image, 0, self.translation_step)
            mask = imutils.translate(mask, 0, self.translation_step)

        elif(key == ord("s")):
            image = imutils.translate(image, 0, -self.translation_step)
            mask = imutils.translate(mask, 0, -self.translation_step)

        if(key == ord("r")):
            image = imutils.rotate(image, self.rotation_step)
            mask = imutils.rotate(mask, self.rotation_step)

        elif(key == ord("e")):
            image = imutils.rotate(image, -self.rotation_step)
            mask = imutils.rotate(mask, -self.rotation_step)

        return image, mask

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
                bird_mask, self.renderer.h_segmentation,
                (self.camera_x_size, self.camera_y_size),
                flags=self.interpolation)
        camera_image = cv2.warpPerspective(
                bird_image, self.renderer.h_segmentation,
                (self.camera_x_size, self.camera_y_size),
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
        bird_mask[np.logical_and(
                overlay_mask >= self.overlay_mask_value -
                self.annotation_buffer,
                overlay_mask <= self.overlay_mask_value +
                self.annotation_buffer)] = self.overlay_mask_value

        bird_img = np.clip(bird_img + overlay_img, 0, 255)

        camera_img, camera_mask = self.get_camera_view(bird_img, bird_mask)

        return bird_img, bird_mask, camera_img, camera_mask

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

        camera_mask_resized = cv2.resize(camera_img, (1000, 1000),
                                         interpolation=self.interpolation)
        camera_img_resized = cv2.resize(camera_mask, (1000, 1000),
                                        interpolation=self.interpolation)

        mask = np.concatenate([bird_mask_resized, camera_mask_resized], axis=0)
        image = np.concatenate([bird_img_resized, camera_img_resized], axis=0)
        result = np.concatenate([mask, image], axis=1)

        cv2.imshow("result", result.astype(np.uint8))
        key = cv2.waitKey(0)
        if key == ord(" "):
            cv2.imwrite(self.mask_write_path + str(index) +
                        ".png", camera_mask)
            cv2.imwrite(self.img_write_path + str(index) +
                        ".png", camera_img)

        return key
