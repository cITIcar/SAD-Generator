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
    """

    def __init__(self):
        """
        Define class attributes.
        """
        self.x = 0

    def draw_startline(self, start_line_rows, start_line_colums, patch_size):
        """
        Draw the basic shape of the startline.

        Parameters:
            start_line_rows : int
                Number of rows that the start line has.
            start_line_colums : int
                Number of colums that the start line has.
            patch_size : int
                Amount of pixels that sedcribe the height and width of a
                single square.

        Return:
            start_line_image : int array
                Synthetic image that represents a real world start line
            start_line_mask : int array
                Annotation of the start line
        """

        white_patch = np.random.randint(150, 255, (patch_size, patch_size))
        start_line_image = np.random.randint(0, 10, (
                patch_size*start_line_rows, patch_size*start_line_colums))
        start_line_mask = np.ones((
                patch_size*start_line_rows, patch_size*start_line_colums))*255

        for i in range(start_line_rows):
            y_start = i * patch_size
            y_end = (i + 1)*patch_size

            for j in range(start_line_colums):
                if 2*j+1+i % 2 > start_line_colums:
                    break
                x_start = (2*j+i % 2)*patch_size
                x_end = (2*j+1+i % 2)*patch_size
                start_line_image[y_start:y_end, x_start:x_end] = white_patch

        return start_line_image, start_line_mask

    def insert_startline(self, zeros_image, zeros_mask, interpolation, k):
        """
        Insert the startline in a large image.

        The large image can be later transformed in camera perspective.
        The startline can be translated and rotated inside the main image.

        Parameters:

        Return:

        """
        if(k == ord("d")):
            zeros_image = imutils.translate(zeros_image, translation_step, 0)
            zeros_mask = imutils.translate(zeros_mask, translation_step, 0)

        elif(k == ord("a")):
            zeros_image = imutils.translate(zeros_image, -translation_step, 0)
            zeros_mask = imutils.translate(zeros_mask, -translation_step, 0)

        if(k == ord("s")):
            zeros_image = imutils.translate(zeros_image, 0, translation_step)
            zeros_mask = imutils.translate(zeros_mask, 0, translation_step)

        elif(k == ord("w")):
            zeros_image = imutils.translate(zeros_image, 0, -translation_step)
            zeros_mask = imutils.translate(zeros_mask, 0, -translation_step)

        if(k == ord("r")):
            zeros_image = imutils.rotate(zeros_image, rotation_step)
            zeros_mask = imutils.rotate(zeros_mask, rotation_step)

        elif(k == ord("e")):
            zeros_image = imutils.rotate(zeros_image, -rotation_step)
            zeros_mask = imutils.rotate(zeros_mask, -rotation_step)

        return zeros_image, zeros_mask

    def get_birds_eye_view(self, render_object, camera_image, camera_mask,
                           interpolation):
        """
        Transform the perspective of the camera images into bird's-eye-view

        Parameters:
            render_object
            camera_image
            camera_mask
            interpolation
        Return:
            bird_image
            bird_mask
        """
        zeros_1 = np.zeros((128, 128))
        zeros_2 = np.zeros((128, 128))
        zeros_1[64:128, :] = camera_mask
        zeros_2[64:128, :] = camera_image
        zeros_1_large = cv2.resize(zeros_1, (640, 480),
                                   interpolation=interpolation)
        zeros_2_large = cv2.resize(zeros_2, (640, 480),
                                   interpolation=interpolation)
        bird_mask = cv2.warpPerspective(zeros_1_large,
                                        inv(render_object.h_segmentation),
                                        (3000, 3000), flags=interpolation)
        bird_image = cv2.warpPerspective(zeros_2_large,
                                         inv(render_object.h_segmentation),
                                         (3000, 3000), flags=interpolation)
        return bird_image, bird_mask

    def get_camera_view(self, render_object, bird_image, bird_mask,
                        interpolation):
        """
        Transform the perspective of bird's-eye-view images into camera view

        Parameters:
            render_object
            bird_image
            bird_mask
            interpolation
        Return:
            camera_image
            camera_mask
        """
        camera_mask_large = cv2.warpPerspective(bird_mask,
                                                render_object.h_segmentation,
                                                (640, 480),
                                                flags=interpolation)
        camera_image_large = cv2.warpPerspective(bird_image,
                                                 render_object.h_segmentation,
                                                 (640, 480),
                                                 flags=interpolation)
        camera_mask = cv2.resize(camera_mask_large, (128, 128),
                                 interpolation=interpolation)
        camera_image = cv2.resize(camera_image_large, (128, 128),
                                  interpolation=interpolation)
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

        bird_mask = np.clip(bird_mask + startline_mask, 0, 255)
        bird_image = np.clip(bird_image + startline_image, 0, 255)

        camera_image, camera_mask = self.get_camera_view(
                r, bird_image, bird_mask, interpolation)

        bird_mask = cv2.resize(bird_mask, (1000, 1000),
                               interpolation=interpolation)
        bird_image = cv2.resize(bird_image, (1000, 1000),
                                interpolation=interpolation)
        bird_mask = bird_mask[0:500, :]
        bird_image = bird_image[0:500, :]

        camera_mask = cv2.resize(camera_mask, (1000, 500),
                                 interpolation=interpolation)
        camera_image = cv2.resize(camera_image, (1000, 500),
                                  interpolation=interpolation)

        mask = np.concatenate([bird_mask, camera_mask], axis=0)
        image = np.concatenate([bird_image, camera_image], axis=0)
        result = np.concatenate([mask, image], axis=1)

        cv2.imshow("result", result.astype(np.uint8))
        key = cv2.waitKey(0)
        return key


if __name__ == "__main__":

    # TODO Overlay annotation only over road and not over background

    with open('config1.json', 'r') as f:
        json_file = f.read()
    config_json = json.loads(json_file)

    translation_step = 25
    scale_step = 25
    rotation_step = 15

    # Define render objects
    c = config.Config("config1.json", debug=False)
    r = render.Renderer(c)
    r.update_position((1400, 0), math.pi)
    interpolation = cv2.INTER_LINEAR  # cv2.INTER_NEAREST
    angle = 0
    offset_x = 1000
    offset_y = 1000
    scale = 1
    k = 0
    forward = False
    startline = Startline()

    start_line_rows = config_json["start_line_rows"]
    start_line_colums = config_json["start_line_colums"]
    patch_size = config_json["patch_size"]

    start_line_image, start_line_mask = startline.draw_startline(
            start_line_rows, start_line_colums, patch_size)

    annotations_list = glob.glob("./real_dataset/annotations/set_*/*.jpg")

    if len(annotations_list) == 0:
        print("no annotated images found under the path")

    for mask_path in annotations_list:
        image_path = mask_path.replace("annotations", "images")

        camera_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        camera_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        bird_image, bird_mask = startline.get_birds_eye_view(
                r, camera_image, camera_mask, interpolation)
        zeros_image = np.zeros((3000, 3000))
        zeros_mask = np.zeros((3000, 3000))

        zeros_image[offset_x:
                    offset_x + patch_size*start_line_rows,
                    offset_y:
                    offset_y + patch_size*start_line_colums] = start_line_image
        zeros_mask[offset_x:offset_x+patch_size*start_line_rows,
                   offset_y:offset_y+patch_size*start_line_colums] = 250

        key = 0
        while key != ord(" ") or ord("s"):
            zeros_image, zeros_mask = startline.insert_startline(
                    zeros_image, zeros_mask, interpolation, key)
            key = startline.visualize_startline(
                    bird_mask, bird_image, camera_image, camera_mask,
                    zeros_image, zeros_mask, interpolation)
            if key == ord("s"):
                cv2.imwrite("camera_mask.png", camera_mask)
                cv2.imwrite("camera_image.png", camera_image)
                break
            if key == ord(" "):
                break
            if key == 27:
                exit()
