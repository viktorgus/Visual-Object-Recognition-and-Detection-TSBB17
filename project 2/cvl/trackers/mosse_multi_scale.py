import numpy as np
import cv2
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

from ..image_io import crop_patch


class MOSSEMultiChannelScale:
    def __init__(self, normalize, learning_rate=0.1, lam=0.01):
        self.template_size = None
        self.last_response = None
        self.region = None
        self.region_shape = None
        self.region_center = None
        self.learning_rate = learning_rate
        self.lam = lam
        self.norm = normalize

        self.A = None
        self.B = None
        self.M = None

    def crop_patch(self, image, region=None):
        if not region:
            region = self.region
        crop = crop_patch(image, region)
        if crop is None:
            return self.region
        else:
            return crop

    def scale_regions(self):
        region_pyramid = []

        for i in range(-10, 11):
            scale_factor = 1.02 ** i
            region_pyramid += [self.region.rescale(scale_factor, True)]

        return region_pyramid

    def scale_estimation(self, image):
        region_pyramid = self.scale_regions()
        responses = [0] * len(region_pyramid)
        for i, region in enumerate(region_pyramid):
            for ch in range(image.shape[2]):
                patch = self.crop_patch(image[:, :, ch], region)
                patch = np.resize(patch, (self.template_size))
                if self.norm:
                    patch = patch / 255
                    patch = patch - np.mean(patch)
                    patch = patch / np.std(patch)
                patch = fft2(patch)
                responses[i] += ifft2(self.M[ch] * np.conj(patch))
        best_i = np.argmax([np.max(resp) for resp in responses])
        self.region = region_pyramid[best_i]
        self.region_center = (self.region.height // 2, self.region.width // 2)

    def start(self, image, region):
        self.region = region
        self.region_shape = (region.height, region.width)
        self.region_center = (region.height // 2, region.width // 2)
        self.template_size = self.region_shape
        c = np.outer(
            np.exp(-((   (np.linspace(-10, 10, num=self.region_shape[0]) -20/self.region_shape[0]) / 2.0) ** 2) / 2),
            np.exp(-((   (np.linspace(-10, 10, num=self.region_shape[1]) -20/self.region_shape[1]) / 2.0) ** 2) / 2),
        )

        self.C = fft2(c)

        patches = []
        for ch in range(image.shape[2]):
            patch = self.crop_patch(image[:, :, ch])
            if self.norm:
                patch = patch / 255
                patch = patch - np.mean(patch)
                patch = patch / np.std(patch)
            patch = patch / np.std(patch)
            patches += [fft2(patch)]
        self.P = patches

        self.A = [np.conj(self.C) * p for p in self.P]
        self.B = sum([np.conj(p) * p for p in self.P])
        self.M = [a / (self.B + self.lam) for a in self.A]

    def detect(self, image):
        responses = []
        for ch in range(image.shape[2]):
            patch = self.crop_patch(image[:, :, ch])
            patch = np.resize(patch, (self.template_size))
            if self.norm:
                patch = patch / 255
                patch = patch - np.mean(patch)
                patch = patch / np.std(patch)
            patch = fft2(patch)
            responses += [ifft2(self.M[ch] * np.conj(patch))]

        response = sum(responses)
        r, c = np.unravel_index(np.argmax(response), response.shape)

        # Keep for visualisation
        self.last_response = response

        r_offset = (
            r - self.region_center[0]
        )
        c_offset = (
            c - self.region_center[1]
        )
        print(f"r: {r}  c: {c}  r_off: {r_offset}  c_off: {c_offset}")
        self.region.xpos += c_offset
        self.region.ypos += r_offset

        self.scale_estimation(image)

        return self.region

    def update(self, image, lr=0.1):
        patches = []
        for ch in range(image.shape[2]):
            patch = self.crop_patch(image[:, :, ch])
            patch = np.resize(patch, (self.template_size))
            if self.norm:
                patch = patch / 255
                patch = patch - np.mean(patch)
                patch = patch / np.std(patch)
            patches += [fft2(patch)]

        self.A = [
            lr * np.conj(self.C) * patches[i] + (1 - lr) * self.A[i]
            for i in range(len(patches))
        ]

        self.B = lr * sum([np.conj(p) * p for p in patches]) + (1 - lr) * self.B
        self.M = [a / (self.B + self.lam) for a in self.A]
