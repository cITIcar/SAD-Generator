import cv2
import numpy as np

from . import render_object

class DrunkDriving(render_object.RenderObject):
    def __init__(self, frequency_from=1, frequency_to=4, max_error_x=200, max_error_y=100, **kwargs):
        self.frequency_from = frequency_from
        self.frequency_to = frequency_to
        self.max_error_x = max_error_x
        self.max_error_y = max_error_y
        self.idx = 0
        self.new_drive_curve()


    def new_drive_curve(self):
        frequency_x = np.random.randint(self.frequency_from, self.frequency_to)
        frequency_y = np.random.randint(self.frequency_from, self.frequency_to)
        error_x = np.random.randint(0, self.max_error_x)
        error_y = np.random.randint(0, self.max_error_y)
        self.drive_curve_x = np.sin(np.linspace(0, np.pi * 2 * frequency_x, 100)) * error_x
        self.drive_curve_y = np.sin(np.linspace(0, np.pi * 2 * frequency_y, 100)) * error_y


    def update_position_step(self, position, angle, **kwargs):
        error_x = self.drive_curve_x[self.idx]
        error_y = self.drive_curve_y[self.idx]
        self.idx += 1
        if self.idx >= len(self.drive_curve_x):
            self.new_drive_curve()
            self.idx = 0

        error_vec = np.array([error_x * np.cos(angle), error_x * np.sin(angle)]) + np.array([error_y * np.sin(angle), error_y * np.cos(angle)])
        position += error_vec

        return position, angle

