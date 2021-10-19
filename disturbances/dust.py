import cv2
import numpy as np

from . import disturbance

class Dust(disturbance.Disturbance):
    def __init__(self, particles=10000, color=255, **kwargs):
        self.particles = particles
        self.color = color


    def pre_transform_step(self, image, **kwargs):
        dust_idcs = (
            np.random.randint(0, image.shape[0], self.particles),
            np.random.randint(0, image.shape[1], self.particles))
        image[dust_idcs] = self.color

