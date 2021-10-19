import random
import numpy as np
import cv2

from . import disturbance

class HiddenGroundRect(disturbance.Disturbance):
    def __init__(self, min_size=1000, max_size=3000, **kwargs):
        self.min_size = min_size
        self.max_size = max_size


    def pre_transform_step(self, image, **kwargs):
        width = random.random() * (self.max_size - self.min_size) + self.min_size
        length = random.random() * (self.max_size - self.min_size) + self.min_size
        self.height = random.random() * 100 + 50
        position_x = random.random() * 1000 + 2580
        position_y = random.random() * 3000 + 3000

        missing_rect = np.array([
            [ position_x + width, position_y - length ],
            [ position_x, position_y - length ],
            [ position_x, position_y ],
            [ position_x + width, position_y ]])

        cv2.fillPoly(image, [missing_rect.astype(int)], 0)

