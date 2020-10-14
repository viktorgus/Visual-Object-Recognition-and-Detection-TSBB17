import pytest


def test_crop_shape():
    from cvl.image_io import crop_patch
    from cvl.dataset import BoundingBox
    import numpy as np

    img = np.random.uniform(size=(1200, 600))
    region = BoundingBox("tl-size", 50, 60, 100, 200)

    crop = crop_patch(img, region)
    assert crop.shape == region.shape()


def test_crop_region():
    from cvl.image_io import crop_patch
    from cvl.dataset import BoundingBox
    import numpy as np

    img = np.random.uniform(size=(1200, 600))
    region = BoundingBox("tl-size", 60, 50, 100, 200)
    crop = crop_patch(img, region)

    assert crop[0, 0] == img[region.ypos, region.xpos]
    assert (
        crop[-1, -1]
        == img[region.ypos + region.height - 1, region.xpos + region.width - 1]
    )


def test_crop_pad_shape_negative():
    from cvl.image_io import crop_patch
    from cvl.dataset import BoundingBox
    import numpy as np

    img = np.random.uniform(size=(1200, 600))
    region = BoundingBox("tl-size", -10, -50, 100, 75)

    crop = crop_patch(img, region)

    assert crop.shape == region.shape()


def test_crop_pad_shape_positive():
    from cvl.image_io import crop_patch
    from cvl.dataset import BoundingBox
    import numpy as np

    img = np.random.uniform(size=(1200, 600))
    region = BoundingBox("tl-size", 1150, 75, 100, 75)

    crop = crop_patch(img, region)

    assert crop.shape == region.shape()


def test_crop_pad_value_negative():
    from cvl.image_io import crop_patch
    from cvl.dataset import BoundingBox
    import numpy as np

    img = np.random.uniform(size=(1200, 600))
    region = BoundingBox("tl-size", -1, 0, 100, 75)

    crop = crop_patch(img, region)

    assert crop[0, 0] == 0
