"""
Augmentation of real world images with start lines.
This GUI allows to transform real world camera images in bird's-eye-view
and add a start line inside it. After the adding of the startline, the image
will be transformed back into camera perspective.
"""

import numpy as np
import cv2
import json
import math
import config
import render
import glob
import imutils
from numpy.linalg import inv


class Startline:
    """
    This class is made for the augmentation of real world images with start
    line images.

    Attributes:
    translation_step : int
        How many pixels the startline moves every time the key is pressed
    scale_step : int
    rotation_step : int
        How many degrees the startline rotates every time the key is pressed
    start_line_rows : int
        Number of rows that the startline has.
    start_line_colums : int
        Number of colums that the startline has.
    patch_size : int
        Amount of pixels that sedcribe the height and width of a
        single square.
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

    def __init__(self):
        """
        Define class attributes.
        """
        with open('config1.json', 'r') as f:
            json_file = f.read()
        startline_config = json.loads(json_file)["startline_config"]
        self.translation_step = startline_config["translation_step"]
        self.scale_step = startline_config["scale_step"]
        self.rotation_step = startline_config["rotation_step"]
        self.start_line_rows = startline_config["start_line_rows"]
        self.start_line_colums = startline_config["start_line_colums"]
        self.patch_size = startline_config["patch_size"]
        self.angle = startline_config["angle"]
        self.offset_x = startline_config["offset_x"]
        self.offset_y = startline_config["offset_y"]
        self.mask_write_path = startline_config["mask_write_path"]
        self.img_write_path = startline_config["img_write_path"]

        # Define render objects
        render_config = config.Config("config1.json", debug=False)
        self.renderer = render.Renderer(render_config)
        self.renderer.update_position((1400, 0), math.pi)
        self.interpolation = cv2.INTER_NEAREST

    def draw_startline(self):
        """
        Draw the basic shape of the startline.

        Parameters:
        None.

        Return:
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
        start_line_mask = np.ones((self.patch_size*self.start_line_rows,
                                   self.patch_size*self.start_line_colums))*250

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

    def insert_startline(self, zeros_image, zeros_mask, k):
        """
        Translate or Rotate image and annotation of startline.

        Depending of the key pressed by the user the startline is translated
        or rotated with the value defined in the config file.

        Parameters:
            zeros_image : numpy array
                Black image that contains the image of the startline
            zeros_mask : numpy array
                Black image that contains the annotation of the startline
            k : int
                int that contains the value of the pressed key

        Return:
            zeros_image : numpy array
                Black image that contains the altered image of the startline
            zeros_mask : numpy array
                Black image that contains the altered annotation of the
                startline
        """
        if(k == ord("a")):
            zeros_image = imutils.translate(zeros_image,
                                            self.translation_step, 0)
            zeros_mask = imutils.translate(zeros_mask,
                                           self.translation_step, 0)

        elif(k == ord("d")):
            zeros_image = imutils.translate(zeros_image,
                                            -self.translation_step, 0)
            zeros_mask = imutils.translate(zeros_mask,
                                           -self.translation_step, 0)

        if(k == ord("w")):
            zeros_image = imutils.translate(zeros_image, 0,
                                            self.translation_step)
            zeros_mask = imutils.translate(zeros_mask, 0,
                                           self.translation_step)

        elif(k == ord("s")):
            zeros_image = imutils.translate(zeros_image, 0,
                                            -self.translation_step)
            zeros_mask = imutils.translate(zeros_mask, 0,
                                           -self.translation_step)

        if(k == ord("r")):
            zeros_image = imutils.rotate(zeros_image, self.rotation_step)
            zeros_mask = imutils.rotate(zeros_mask, self.rotation_step)

        elif(k == ord("e")):
            zeros_image = imutils.rotate(zeros_image, -self.rotation_step)
            zeros_mask = imutils.rotate(zeros_mask, -self.rotation_step)

        return zeros_image, zeros_mask

    def get_birds_eye_view(self, camera_image, camera_mask):
        """
        Transform the perspective of the camera images into bird's-eye-view.

        Parameters:
            camera_image : numpy array
                Image in camera perspective
            camera_mask : numpy array
                Annotation in camera perspective
        Return:
            bird_image : numpy array
                Image in bird's-eye-view
            bird_mask : numpy array
                Annotation in bird's-eye-view
        """
        zeros_1 = np.zeros((128, 128))
        zeros_2 = np.zeros((128, 128))
        zeros_1[64:128, :] = camera_mask
        zeros_2[64:128, :] = camera_image
        zeros_1_large = cv2.resize(zeros_1, (640, 480),
                                   interpolation=self.interpolation)
        zeros_2_large = cv2.resize(zeros_2, (640, 480),
                                   interpolation=self.interpolation)
        bird_mask = cv2.warpPerspective(
                zeros_1_large, inv(self.renderer.h_segmentation),
                (3000, 3000), flags=self.interpolation)
        bird_image = cv2.warpPerspective(
                zeros_2_large, inv(self.renderer.h_segmentation),
                (3000, 3000), flags=self.interpolation)
        return bird_image, bird_mask

    def get_camera_view(self, bird_image, bird_mask):
        """
        Transform the perspective of bird's-eye-view images into camera view.

        Parameters:
            bird_image : numpy array
                Image in bird's-eye-view
            bird_mask : numpy array
                Annotation in bird's-eye-view
        Return:
            camera_image : numpy array
                Image in camera perspective
            camera_mask : numpy array
                Annotation in camera perspective
        """
        camera_mask_large = cv2.warpPerspective(
                bird_mask, self.renderer.h_segmentation, (640, 480),
                flags=self.interpolation)
        camera_image_large = cv2.warpPerspective(
                bird_image, self.renderer.h_segmentation, (640, 480),
                flags=self.interpolation)
        camera_mask = cv2.resize(camera_mask_large, (128, 128),
                                 interpolation=self.interpolation)
        camera_image = cv2.resize(camera_image_large, (128, 128),
                                  interpolation=self.interpolation)
        camera_mask = camera_mask[64:128, :]
        camera_image = camera_image[64:128, :]
        return camera_image, camera_mask

    def visualize_startline(self, bird_mask, bird_image, camera_image,
                            camera_mask, startline_image, startline_mask,
                            interpolation):
        """
        Visualize the mask and image of the bird's-eye-view and camera view.

        Parameters:
            bird_mask
            bird_image
            camera_image
            camera_mask
            startline_image
            startline_mask
            interpolation

        Return:
            key
        """

        # Delete the startline where there is no road below it
        startline_image[np.logical_or(bird_mask >= 175, bird_mask <= 25)] = 0

        # Only annotate startline where there was road below it
        startline_mask[np.logical_or(bird_mask >= 175, bird_mask <= 25)] = 0

        bird_mask[np.logical_and(
                np.logical_and(bird_mask <= 175, startline_mask >= 225),
                bird_mask >= 25)] = 250

        bird_image = np.clip(bird_image + startline_image, 0, 255)

        startline_camera_image, startline_camera_mask = self.get_camera_view(
                startline_image, startline_mask)

        camera_mask_ = np.clip(startline_camera_mask + camera_mask, 0, 255)
        camera_image_ = np.clip(startline_camera_image + camera_image, 0, 255)

        bird_mask = cv2.resize(bird_mask, (1000, 1000),
                               interpolation=interpolation)
        bird_image = cv2.resize(bird_image, (1000, 1000),
                                interpolation=interpolation)

        bird_mask = cv2.resize(bird_mask, (1000, 1000),
                               interpolation=interpolation)
        bird_image = cv2.resize(bird_image, (1000, 1000),
                                interpolation=interpolation)
        bird_mask = bird_mask[0:500, :]
        bird_image = bird_image[0:500, :]

        camera_mask = cv2.resize(camera_mask_, (1000, 500),
                                 interpolation=interpolation)
        camera_image = cv2.resize(camera_image_, (1000, 500),
                                  interpolation=interpolation)

        mask = np.concatenate([bird_mask, camera_mask], axis=0)
        image = np.concatenate([bird_image, camera_image], axis=0)
        result = np.concatenate([mask, image], axis=1)

        cv2.imshow("result", result.astype(np.uint8))
        key = cv2.waitKey(0)
        if key == ord(" "):
            cv2.imwrite(self.mask_write_path + str(index) +
                        ".png", camera_mask_)
            cv2.imwrite(self.img_write_path + str(index) +
                        ".png", camera_image_)
        return key


if __name__ == "__main__":

    # TODO save image
    # TODO create some UI

    startline = Startline()
    index = 0

    start_line_image, start_line_mask = startline.draw_startline()

    annotations_list = glob.glob("./real_dataset/annotations/set_*/*.jpg")

    if len(annotations_list) == 0:
        print("no annotated images found under the path")

    for mask_path in annotations_list:
        image_path = mask_path.replace("annotations", "images")

        camera_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        camera_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        bird_image, bird_mask = startline.get_birds_eye_view(
                camera_image, camera_mask)
        zeros_image = np.zeros((3000, 3000))
        zeros_mask = np.zeros((3000, 3000))

        zeros_image[startline.offset_x:
                    startline.offset_x +
                    startline.patch_size*startline.start_line_rows,
                    startline.offset_y:
                    startline.offset_y +
                    startline.patch_size*startline.start_line_colums] = start_line_image
        zeros_mask[startline.offset_x:startline.offset_x +
                   startline.patch_size*startline.start_line_rows,
                   startline.offset_y:startline.offset_y +
                   startline.patch_size*startline.start_line_colums] = 250

        key = 0
        while key != ord(" ") or ord("q"):
            zeros_image, zeros_mask = startline.insert_startline(
                    zeros_image, zeros_mask, key)
            key = startline.visualize_startline(
                    np.copy(bird_mask), np.copy(bird_image), camera_image,
                    np.copy(camera_mask), np.copy(zeros_image),
                    np.copy(zeros_mask), startline.interpolation)
            if key == ord(" "):
                break
            if key == ord("q"):
                break
            if key == 27:
                exit()
        index += 1
