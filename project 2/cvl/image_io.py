import numpy as np
from PIL import Image


def crop_patch(image, region):
    """
    Crop an image patch with padding
    Patch is guaranteed to have the same size as the region.
    If the region is partially outside the image, zeros will be added at the edge
    If the region is completely outside the image it will crash or missbehave
    """

    r0 = region.ypos
    r1 = region.ypos + region.height
    c0 = region.xpos
    c1 = region.xpos + region.width

    ri0 = r0
    ri1 = r1
    rp0 = 0
    rp1 = region.height

    ci0 = c0
    ci1 = c1
    cp0 = 0
    cp1 = region.width

    if c0 < 0:
        ci0 = 0
        cp0 = -c0

    if r0 < 0:
        ri0 = 0
        rp0 = -r0

    if r1 >= image.shape[0]:
        ri1 = image.shape[0]
        rp1 = region.height - (r1 - image.shape[0])

    if c1 >= image.shape[1]:
        ci1 = image.shape[1]
        cp1 = region.width - (c1 - image.shape[1])

    patch = np.zeros(shape=(region.height, region.width), dtype=image.dtype)

    patch[rp0:rp1, cp0:cp1] = image[ri0:ri1, ci0:ci1]

    if not patch.shape == (region.height, region.width):
        return None
    return patch


def read_image(image_path, transform=None):
    file_ending = image_path.suffix

    pillow_valid_endings = [".png", ".jpg", ".jpeg", ".webp"]

    if file_ending == ".pfm":
        pixel_array = read_flow(image_path)
    elif file_ending == ".npy":
        pixel_array = np.load(image_path)
    elif file_ending in pillow_valid_endings:
        pixel_array = np.asarray(Image.open(image_path))
    else:
        raise NotImplementedError(
            "No way to read files of type: {}".format(file_ending)
        )

    if transform is not None:
        pixel_array = transform(pixel_array)

    return np.ascontiguousarray(pixel_array)


def write_image(image_path, pixels):
    file_ending = image_path.suffix

    pillow_valid_endings = [".png", ".jpg", ".jpeg", ".webp"]

    if file_ending in pillow_valid_endings:
        Image.fromarray(pixels).save(image_path)
    else:
        raise ValueError("Invalid file ending, does not know how to write this")
