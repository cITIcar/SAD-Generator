import numpy as np
import cv2
from numpy import cos, sin
import math

from . import render_object

class GaussReflection(render_object.RenderObject):
    def __init__(self, position):
        self.gauss_reflection = GaussReflection.create_gauss_reflection()
        self.position = position


    def post_transform_step(self, image, global_angle, bird_to_camera_segment, **kwargs):
        self._add_reflection(image, global_angle, bird_to_camera_segment)


    def create_gauss_reflection():
        """
        create a Gauss curve, intended to be used as a reflection
        """
        x, y = np.meshgrid(np.linspace(-2, 2, 1000), np.linspace(-1, 1, 1000))
        dst = np.sqrt(x * x + y * y)
        sigma = 0.01
        gauss = np.exp(-(dst ** 10) / (2 * sigma ** 2))

        return (gauss * 127).astype(np.uint8)


    def _add_reflection(self, image, angle, bird_to_camera_segment):
        """
        add a reflection of an infinitely far away light source
        """
        angle = angle / 180 * math.pi
        angle = -angle
        b = bird_to_camera_segment
        b = b @ np.array([
            [cos(angle), -sin(angle), 0],
            [sin(angle), cos(angle), 0],
            [0, 0, 1]])
        b = b @ np.array([
            [1, 0, -self.position[0]],
            [0, 1, -self.position[1]],
            [0, 0, 1]])
        perspective = cv2.warpPerspective(self.gauss_reflection, b, (640, 480), flags=cv2.INTER_NEAREST)
        perspective[:int(len(perspective) * 0.5)] = 0
        image += perspective

