import random
import numpy as np
import cv2
import time

from . import disturbance
from render import Renderer

class BoxObstacle(disturbance.Disturbance):
    def __init__(self, min_size=50, max_size=150, colors=[200, 127, 100, 50], config=None):
        self.min_size = min_size
        self.max_size = max_size
        self.colors = colors
        self.label_color = config["obstacle_class"]
        self.rescale = config["rescale"] if config else 1
        self.blur = config["output_size"][0] // 50
        if self.blur % 2 == 0:
            self.blur += 1
        self.camera_height_px = config["camera_height"] * config["px_per_cm"]

    def pre_transform_step(self, **kwargs):
        """
        create random rectangle on the ground
        """
        width = random.random() * (self.max_size - self.min_size) + self.min_size
        length = random.random() * (self.max_size - self.min_size) + self.min_size
        self.height = random.random() * 100 + 50
        position_x = random.random() * 1000 + 2580
        position_y = random.random() * 3000 + 3000

        self.ground = np.array([
            [ position_x + width, position_y - length ],
            [ position_x, position_y - length ],
            [ position_x, position_y ],
            [ position_x + width, position_y ]])

        self.points = np.array([
            [*self.ground[0], 0],
            [*self.ground[0], self.height],
            [*self.ground[1], self.height],
            [*self.ground[1], 0],
            [*self.ground[2], 0],
            [*self.ground[2], self.height],
            [*self.ground[3], self.height],
            [*self.ground[3], 0]])

        self.surfaces_reflection = [
            np.array([
                self.points[2],
                self.points[3],
                self.points[4],
                self.points[5]]) * np.array([1, 1, -1]),

            np.array([
                self.points[0],
                self.points[1],
                self.points[6],
                self.points[7]]) * np.array([1, 1, -1]),

            np.array([
                self.points[4],
                self.points[5],
                self.points[6],
                self.points[7]]) * np.array([1, 1, -1]),

            np.array([
                self.points[1],
                self.points[2],
                self.points[5],
                self.points[6]]) * np.array([1, 1, -1])]

        self.surfaces = [
            np.array([
                self.points[2],
                self.points[3],
                self.points[4],
                self.points[5]]),

            np.array([
                self.points[0],
                self.points[1],
                self.points[6],
                self.points[7]]),

            np.array([
                self.points[4],
                self.points[5],
                self.points[6],
                self.points[7]]),

            np.array([
                self.points[1],
                self.points[2],
                self.points[5],
                self.points[6]])]

        self.ordering = self.ground[2][1]


    def post_transform_step(self, image, image_label, point, angle, bird_to_camera_nice, bird_to_camera_label, renderer, **kwargs):
        diff = self.points[:,:2] - np.repeat([np.array(point)], 8, axis=0)
        rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])

        if not np.all((diff @ rot)[:,1] < 0):
            return

        reflection = self.create_reflection(image, image_label, point, angle, bird_to_camera_nice, bird_to_camera_label, renderer)
        image += cv2.resize(reflection, image.shape[::-1], interpolation=cv2.INTER_NEAREST)
        self.create_obstacle(image, image_label, point, angle, bird_to_camera_nice, bird_to_camera_label, renderer)


    def create_reflection(self, image, image_label, point, angle, bird_to_camera_nice, bird_to_camera_label, renderer):
        empty = np.zeros(image_label.shape)

        self.surfaces_reflection.sort(key=lambda surface: -np.linalg.norm(np.average(surface, axis=0) - np.array([*point, self.camera_height_px])))

        for surface, color in zip(self.surfaces_reflection, self.colors):
            points_2d = np.array(list(map(
                lambda p: renderer.project_point(p),
                surface)))

            cv2.fillPoly(empty, [(points_2d / self.rescale).astype(int)], int(color / 1.5))
            empty = cv2.blur(empty, (self.blur, self.blur))

        return empty


    def create_obstacle(self, image, image_label, point, angle, bird_to_camera_nice, bird_to_camera_label, renderer):
        self.surfaces.sort(key=lambda surface: -np.linalg.norm(np.average(surface, axis=0) - np.array([*point, self.camera_height_px])))

        for surface, color in zip(self.surfaces, self.colors):
            points_2d = np.array(list(map(
                lambda p: renderer.project_point(p),
                surface)))

            cv2.fillPoly(image, [points_2d.astype(int)], color)
            cv2.fillPoly(image_label, [(points_2d / self.rescale).astype(int)], self.label_color)

