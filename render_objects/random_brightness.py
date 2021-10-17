import cv2
import numpy as np

from . import render_object

class RandomBrightness(render_object.RenderObject):
    def post_transform_step(self, image, **kwargs):
        brighten = np.random.randint(0, 30)
        image += brighten
