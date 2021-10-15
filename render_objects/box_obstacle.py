import random
import numpy as np
import cv2

from . import render_object
from render import Renderer

class BoxObstacle(render_object.RenderObject):
    def __init__(self, min_size=50, max_size=150, colors=[200, 127, 100, 50], config=None):
        self.min_size = min_size
        self.max_size = max_size
        self.colors = colors
        self.segment_color = config["obstacle_class"]
        self.rescale = config["rescale"] if config else 1


    def pre_transform_step(self, **kwargs):
        """
        create random rectangle relative to the cars position
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

        self.ordering = self.ground[2][1]


    def post_transform_step(self, image, image_segment, point, angle, bird_to_camera_nice, bird_to_camera_segment, renderer, **kwargs):
        points = np.array([
            [*self.ground[0], 0],
            [*self.ground[0], self.height],
            [*self.ground[1], self.height],
            [*self.ground[1], 0],
            [*self.ground[2], 0],
            [*self.ground[2], self.height],
            [*self.ground[3], self.height],
            [*self.ground[3], 0]])

        if not np.all(points[:,1] < point[1]):
            return

        surfaces = [
            np.array([
                points[2],
                points[3],
                points[4],
                points[5]]),

            np.array([
                points[0],
                points[1],
                points[6],
                points[7]]),

            np.array([
                points[4],
                points[5],
                points[6],
                points[7]]),

            np.array([
                points[1],
                points[2],
                points[5],
                points[6]])]

        surfaces.sort(key=lambda surface: -np.linalg.norm(np.average(surface, axis=0) - np.array([*point, 0])))
 
        for surface, color in zip(surfaces, self.colors):
            points_2d = np.array(list(map(
                lambda p: renderer.project_point(p),
                surface)))

            cv2.fillPoly(image, [points_2d.astype(int)], color)
            cv2.fillPoly(image_segment, [(points_2d / self.rescale).astype(int)], self.segment_color)

