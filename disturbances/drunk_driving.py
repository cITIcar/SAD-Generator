import cv2
import numpy as np

from . import disturbance

class DrunkDriving(disturbance.Disturbance):
    def __init__(self, frequency_from=1, frequency_to=4, max_error_x=200, max_error_y=100, max_error_angle=np.pi / 2, **kwargs):
        self.frequency_from = frequency_from
        self.frequency_to = frequency_to
        self.max_error_x = max_error_x
        self.max_error_y = max_error_y
        self.max_error_angle = max_error_angle
        self.idx = 0
        self.new_drive_curve()


    def new_drive_curve(self):
        frequency_x = np.random.randint(self.frequency_from, self.frequency_to)
        frequency_y = np.random.randint(self.frequency_from, self.frequency_to)
        frequency_angle = np.random.randint(self.frequency_from, self.frequency_to)

        error_x = np.random.randint(0, self.max_error_x)
        error_y = np.random.randint(0, self.max_error_y)
        error_angle = np.random.random() * (self.max_error_angle * 2) - self.max_error_angle

        self.drive_curve_x = np.sin(np.linspace(0, np.pi * 2 * frequency_x, 100)) * error_x
        self.drive_curve_y = np.sin(np.linspace(0, np.pi * 2 * frequency_y, 100)) * error_y
        self.drive_curve_angle = np.sin(np.linspace(0, np.pi * 2 * frequency_angle, 100)) * error_angle


    def update_position_step(self, position, angle, **kwargs):
        error_x = self.drive_curve_x[self.idx]
        error_y = self.drive_curve_y[self.idx]
        error_angle = self.drive_curve_angle[self.idx]
        self.idx += 1
        if self.idx >= len(self.drive_curve_x):
            self.new_drive_curve()
            self.idx = 0

        angle += error_angle
        error_vec = np.array([error_x * np.cos(angle), error_x * np.sin(angle)]) + np.array([error_y * np.sin(angle), error_y * np.cos(angle)])
        position += error_vec

        return position, angle

