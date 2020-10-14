import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import math
from ..image_io import crop_patch


class MOSSETracker:
    def __init__(self,lol, learning_rate=0.1, lam=0.01):
        self.template = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate
        self.lam = lam

        self.A = None
        self.B = None
        self.M = None

    def crop_patch(self, image):
        region = self.region
        crop = crop_patch(image, region)
        if crop is None:
            return self.region
        else:
            return crop

    def start(self, image, region):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        patch = self.crop_patch(image)

        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)

        self.P = fft2(patch)
        c = np.outer(
            np.exp(-((  (np.linspace(-10, 10, num=self.region_shape[0]) -20/self.region_shape[0]  ) / 2.0) ** 2) / 2),
            np.exp(-((  (np.linspace(-10, 10, num=self.region_shape[1]) -20/self.region_shape[1]) / 2.0) ** 2) / 2),
        )

        # h = np.arange(0, self.region_shape[0]) - (self.region_shape[0] - 1) / 2
        # w = np.arange(0, self.region_shape[1]) - (self.region_shape[1] - 1) / 2
        # c = np.outer(
        #     np.exp(-(h ** 2) / 2),
        #     np.exp(-(w ** 2) / 2),
        # )
        # c /= c.max()

        self.C = fft2(c)

        self.A = np.conj(self.C) * self.P
        self.B = np.conj(self.P) * self.P

        self.M = self.A / (self.B + self.lam)

    def detect(self, image):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        P = fft2(patch)

        responsef = self.M * np.conj(P)
        response = ifft2(responsef)

        r, c = np.unravel_index(np.argmax(response), response.shape)
        # Keep for visualisation
        self.last_response = response
        print(f"response dim: {response.shape}")
        print(f"patch dim : {patch.shape}")
        r_offset = (
            r - self.region_center[0]
        )
        c_offset = (
            c - self.region_center[1]
        )
        print(f"r offset: {r_offset}  c offset: {c_offset}")
        self.region.xpos += c_offset
        self.region.ypos += r_offset

        return self.region

    def update(self, image, lr=0.1):
        assert len(image.shape) == 2, "NCC is only defined for grayscale images"
        patch = self.crop_patch(image)
        patch = patch / 255
        patch = patch - np.mean(patch)
        patch = patch / np.std(patch)
        P = fft2(patch)

        self.A = lr * (np.conj(self.C) * P) + (1 - lr) * self.A
        self.B = lr * (np.conj(P) * P) + (1 - lr) * self.B

        self.M = self.A / (self.B + self.lam)