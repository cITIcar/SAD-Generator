import math
import numpy as np
from numpy import sin, cos
import cv2

import config


class Renderer:
    def __init__(self, config):
        self.config = config
        self.angle_x = np.pi / 2 + np.radians(config["camera_angle"])
        self.angle_y = np.radians(0)
        self.angle_z = np.radians(0)

        self.position_x = 0
        self.position_y = 0
        self.position_z = config["camera_height_px"]

        self.render_objects = []
        self.update_position((self.position_x, self.position_y), self.angle_z)


    def project_point(self, coord):
        x, y, z = coord
        coord_3d_h = np.array([-x, -y, -z, 1])
        coord_3d_h = self.translation @ coord_3d_h
        coord_2d_h = self.config["intrinsic_camera_matrix"] @ self.rotation @ coord_3d_h
        coord_2d = coord_2d_h[:2] / coord_2d_h[2]

        return coord_2d


    def update_position(self, position, angle):
        for obj in self.render_objects:
            if obj.update_position_step:
                position, angle = obj.update_position_step(
                    position=position,
                    angle=angle,
                    renderer=self)

        self.position_x, self.position_y = position
        self.angle_z = -angle
        rotation_x = np.array([
            [1, 0, 0, 0],
            [0, cos(self.angle_x), -sin(self.angle_x), 0],
            [0, sin(self.angle_x), cos(self.angle_x), 0],
            [0, 0, 0, 1]])
        rotation_y = np.array([
            [cos(self.angle_y), 0, sin(self.angle_y), 0],
            [0, 1, 0, 0],
            [-sin(self.angle_y), 0, cos(self.angle_y), 0],
            [0, 0, 0, 1]])
        rotation_z = np.array([
            [cos(self.angle_z), -sin(self.angle_z), 0, 0],
            [sin(self.angle_z), cos(self.angle_z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
        self.rotation = rotation_x @ rotation_y @ rotation_z

        self.translation = np.array([
            [1, 0, 0, self.position_x],
            [0, 1, 0, self.position_y],
            [0, 0, 1, self.position_z],
            [0, 0, 0, 1]])

        self.h_camera = self.config["intrinsic_camera_matrix"] @ self.rotation @ self.translation @ self.config["project2d"]
        self.h_segmentation = self.config["intrinsic_segmentation_matrix"] @ self.rotation @ self.translation @ self.config["project2d"]


    def update_ground_plane(self, image_real, image_segmentation, render_objects):
        self.image_real = image_real
        self.image_segmentation = image_segmentation
        self.render_objects = sorted(render_objects, key=lambda obj: obj.ordering)

        for obj in self.render_objects:
            if obj.pre_transform_step:
                obj.pre_transform_step(
                    image=image_real,
                    image_segment=image_segmentation)

        self.render_objects.sort(key=lambda obj: obj.ordering)


    def render_images(self):
        width, height = self.config["output_size"]
        rescale = self.config["rescale"]
        perspective_camera = cv2.warpPerspective(self.image_real, self.h_camera, (width * rescale, height * rescale), flags=cv2.INTER_NEAREST).astype(np.float32)
        perspective_segmentation = cv2.warpPerspective(self.image_segmentation, self.h_segmentation, (width, height), flags=cv2.INTER_NEAREST).astype(np.float32)

        # TODO: calc horizon
        perspective_camera[:int(len(perspective_camera) * 0.45)] = 0
        perspective_segmentation[:int(len(perspective_segmentation) * 0.45)] = 0

        for obj in self.render_objects:
            if obj.post_transform_step:
                obj.post_transform_step(
                    image=perspective_camera,
                    image_segment=perspective_segmentation,
                    point=(self.position_x, self.position_y),
                    angle=self.angle_z,
                    global_angle=self.angle_z, # TODO: track global angle
                    bird_to_camera_nice=self.h_camera,
                    bird_to_camera_segment=self.h_segmentation,
                    renderer=self)

        if rescale != 1:
            perspective_camera = cv2.resize(perspective_camera, (width, height), interpolation=cv2.INTER_AREA)

        return perspective_camera, perspective_segmentation

