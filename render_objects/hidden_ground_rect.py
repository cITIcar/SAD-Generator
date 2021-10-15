import random
import numpy as np
import cv2

from . import render_object

class HiddenGroundRect(render_object.RenderObject):
    def __init__(self, min_size=1000, max_size=3000):
        self.min_size = min_size
        self.max_size = max_size


    def pre_transform_step(self, drive_points, image, **kwargs):
        width = random.random() * (self.max_size - self.min_size) + self.min_size
        length = random.random() * (self.max_size - self.min_size) + self.min_size
        position_x = (random.random() - 0.5) * 2000
        position_y = random.random() * 1500 + 500

        missing_rect = np.array([
            [ drive_points[0][0] + position_x - width, drive_points[0][1] - position_y - length ],
            [ drive_points[0][0] + position_x, drive_points[0][1] - position_y - length ],
            [ drive_points[0][0] + position_x, drive_points[0][1] - position_y ],
            [ drive_points[0][0] + position_x - width, drive_points[0][1] - position_y ]])

        cv2.fillPoly(image, [missing_rect.astype(int)], 0)

