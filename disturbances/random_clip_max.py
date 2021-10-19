import cv2
import numpy as np

from . import disturbance

class RandomClipMax(disturbance.Disturbance):
    def __init__(self, frequency_from=1, frequency_to=20, **kwargs):
        self.frequency_from = frequency_from
        self.frequency_to = frequency_to
        self.idx = 0
        self.new_clip_curve()


    def new_clip_curve(self):
        frequency = np.random.randint(self.frequency_from, self.frequency_to)
        self.brightness_clip = np.sin(np.linspace(0, np.pi * 2 * frequency, 100))


    def post_transform_step(self, image, **kwargs):
        clip = self.brightness_clip[self.idx]
        self.idx += 1
        if self.idx >= len(self.brightness_clip):
            self.new_clip_curve()
            self.idx = 0
        brightness = 155 + clip * 100
        image[image > brightness] = brightness

