import cv2
import numpy as np

from . import disturbance

class RandomBrightness(disturbance.Disturbance):
    def post_transform_step(self, image, **kwargs):
        brighten = np.random.randint(0, 30)
        image += brighten
