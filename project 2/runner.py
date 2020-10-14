#!/usr/bin/env python3
from copy import copy

import math
import cv2
import numpy as np
from tqdm import tqdm
import webcolors

from cvl.dataset import OnlineTrackingBenchmark
from cvl.trackers.ncc_tracker import NCCTracker
from cvl.trackers.mosse_tracker import MOSSETracker
from cvl.trackers.mosse_multi import MOSSEMultiChannel
from cvl.trackers.mosse_multi_scale import MOSSEMultiChannelScale
from cvl.features import alexnetFeatures
from skimage.feature import hog


# from cvl.trackers.mosse_multi import MOSSEMultiChannel
# scales patch to the same sice as imageToMatch by upscaling and padding
def upscale_XY(imageToMatch, patch):
    if (
        patch.shape[0] != imageToMatch.shape[0]
        or patch.shape[1] != imageToMatch.shape[1]
    ):
        X_upscale = math.floor(imageToMatch.shape[0] / patch.shape[0])
        Y_upscale = math.floor(imageToMatch.shape[1] / patch.shape[1])
        upscaled = patch.repeat(X_upscale, axis=0).repeat(Y_upscale, axis=1)

        paddingArr = np.ones(
            (imageToMatch.shape[0], imageToMatch.shape[1], upscaled.shape[2])
        )
        paddingArr[: upscaled.shape[0], : upscaled.shape[1], :] = upscaled
        return paddingArr
    return patch


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def select_features(image, feature_dict, net):
    features = np.ndarray([image.shape[0], image.shape[1], 1])
    if feature_dict["greyscale"]:
        greyscale = np.sum(image, 2) / 3
        features = np.dstack((features, greyscale[:, :, np.newaxis]))
    if feature_dict["rgb"]:
        features = np.dstack((features, image))
    if feature_dict["color_names"]:
        colors = np.empty_like(image, dtype=np.str)
        for a in tqdm(range(image.shape[0])):
            for b in range(image.shape[1]):
                actual, closest = get_colour_name(tuple(image[a, b]))
                colors[a, b] = actual if actual else closest
        print(colors)
    if feature_dict["gradient_filter_bank"]:
        filters = []
        kernel_size = 15

        # Build filters
        for theta in np.arange(0, np.pi / 2, np.pi / 4):
            kern = cv2.getGaborKernel(
                (kernel_size, kernel_size), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F
            )
            kern /= 1.5 * kern.sum()
            filters.append(kern)

        # Apply filters to each channel of the input image
        for kern in filters:
            filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kern)
            features = np.dstack((features, filtered_image))
    if feature_dict["hog"]:
        nBinsX = 40
        nBinsY = 40
        cellSize_X = image.shape[0] / nBinsX
        cellSize_Y = image.shape[1] / nBinsY
        hog_hist = hog(
            image,
            orientations=8,
            pixels_per_cell=(cellSize_X, cellSize_Y),
            cells_per_block=(1, 1),
            multichannel=True,
            feature_vector=False,
        )
        hog_hist = hog_hist[:, :, 0, 0, :]
        upsample_fd = hog_hist.repeat(cellSize_X, axis=0).repeat(cellSize_Y, axis=1)
        features = np.dstack((features, upsample_fd))
    if feature_dict["deep_feats"]:
        for layer in feature_dict["layers"]:
            net_out = net.forward(image, layers=layer)
            upscaled = cv2.resize(net_out, (image.shape[1], image.shape[0]))
            features = np.dstack((features, upscaled))
    features = features[:, :, 1:]

    return features


if __name__ == "__main__":
    feature_dict = {
        "greyscale": False,
        "rgb": True,
        "color_names": False,
        "gradient_filter_bank": True,
        "hog": True,
        "deep_feats": False,
        "layers": [4],  # 1-5
        "normalize": True,
    }
    dataset_path = "otb_mini"

    SHOW_TRACKING = True
    SEQUENCE_IDX = [0]
    dataset = OnlineTrackingBenchmark(SEQUENCE_IDX, dataset_path)

    a_seq = [dataset[i] for i in SEQUENCE_IDX]

    output = []

    net = alexnetFeatures()

    if SHOW_TRACKING:
        cv2.namedWindow("tracker")

    tracker = MOSSEMultiChannel(feature_dict["normalize"])
    for seq_id, seq in enumerate(tqdm(a_seq, desc="Sequence")):
        out = []
        for frame_idx, frame in enumerate(tqdm(seq, desc="Frame")):
            image_color = frame["image"]
            features = select_features(image_color, feature_dict, net)

            if frame_idx == 0:
                bbox = frame["bounding_box"]

                out += [bbox]
                if bbox.width % 2 == 0:
                    bbox.width += 1

                if bbox.height % 2 == 0:
                    bbox.height += 1

                current_position = bbox
                tracker.start(features, bbox)
            else:
                try:
                    tracker.detect(features)
                    tracker.update(features)
                except ValueError:
                    print("fakkkk")
                    break

            out += [copy(tracker.region)]

            if SHOW_TRACKING:
                bbox = tracker.region
                pt0 = (bbox.xpos, bbox.ypos)
                pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
                image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_color, pt0, pt1, color=(0, 255, 0), thickness=3)
                cv2.imshow("tracker", image_color)
                cv2.waitKey(0)
        output += [out]

    # Evaluation

    print(dataset.calculate_performance(output))
