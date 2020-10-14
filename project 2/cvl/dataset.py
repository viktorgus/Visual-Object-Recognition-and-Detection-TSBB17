from pathlib import Path
import os
import math

import matplotlib.patches as patches
import numpy as np
from .image_io import read_image


class BoundingBox(object):
    def __init__(self, *args):
        self.xpos = None
        self.ypos = None
        self.width = None
        self.height = None

        if args[0] == "minmax":
            self.build_minmax(args[1], args[2], args[3], args[4])
        elif args[0] == "tl-size":
            self.build_tlsize(args[1], args[2], args[3], args[4])
        elif args[0] == "np-tl-size":
            a = args[1]
            self.build_tlsize(a[0], a[1], a[2], a[3])
        elif args[0] == "corners":
            self.build_corners(args[1:])
        else:
            raise Exception("Unkown data format: {0}".format(args[0]))

    def shape(self):
        return self.height, self.width

    def rescale(self, scale_factor, round_coordinates=False):
        nw = self.width * scale_factor
        nh = self.height * scale_factor

        cx, cy = self.get_center()

        if round_coordinates:
            nx = math.floor(cx - nw / 2)
            ny = math.floor(cy - nh / 2)
        else:
            nx = cx - nw / 2
            ny = cy - nh / 2

        if round_coordinates:
            return BoundingBox("tl-size", nx, ny, math.floor(nw), math.floor(nh))
        else:
            return BoundingBox("tl-size", nx, ny, nw, nh)

    def get_center(self):
        cx = self.xpos + self.width / 2
        cy = self.ypos + self.width / 2

        return cx, cy

    def as_patch(self, color, linewidth=3, facecolor="none"):
        return patches.Rectangle(
            (self.xpos, self.ypos),
            self.width,
            self.height,
            facecolor=facecolor,
            edgecolor=color,
            linewidth=linewidth,
        )

    def build_corners(self, corners):
        assert len(corners) == 8, "Invalid number of corners, got {}".format(
            len(corners)
        )

        x = corners[0::2]
        y = corners[1::2]

        cx = np.mean(x)
        cy = np.mean(y)

        x1 = min(x)
        x2 = max(x)

        y1 = min(y)
        y2 = max(y)

        s1 = np.linalg.norm(np.array(corners[0:2]) - np.array(corners[2:4]))
        s2 = np.linalg.norm(np.array(corners[2:4]) - np.array(corners[4:6]))

        A1 = s1 * s2
        A2 = (x2 - x1) * (y2 - y1)

        s = np.sqrt(A1 / A2)

        w = s * (x2 - x1)
        h = s * (y2 - y1)

        self.xpos = x1
        self.ypos = y2

        self.width = w
        self.height = h

        assert self.width > 0
        assert self.height > 0

    def intersection_box(self, other):
        """Returns the intersection bb with other box, if it does not exist a box of zero area is returned"""
        x1 = self.xpos
        x2 = other.xpos
        y1 = self.ypos
        y2 = other.ypos

        xx1 = self.xpos + self.width
        xx2 = other.xpos + other.width

        yy1 = self.ypos + self.height
        yy2 = other.ypos + other.height

        ix = max(x1, x2)
        iy = max(y1, y2)

        ixx = min(xx1, xx2)
        iyy = min(yy1, yy2)

        iw = ixx - ix
        ih = iyy - iy

        if iw < 0:
            ix = 0
            iy = 0
            iw = 0
            ih = 0

        if ih < 0:
            ix = 0
            iy = 0
            ih = 0
            ih = 0

        return BoundingBox("tl-size", ix, iy, iw, ih)

    def union_box(self, other):
        """Returns the bounding box covering both this and the other"""
        x1 = min(self.xpos, other.xpos)
        y1 = min(self.ypos, other.ypos)
        x2 = max(x1 + self.width, x1 + other.width)
        y2 = max(y1 + self.height, y1 + other.height)

        width = x2 - x1
        height = y2 - y1

        return BoundingBox("tl-size", x1, y1, width, height)

    def build_BoundingBox2d(self, bb2d):
        self.xpos = bb2d.X()
        self.ypos = bb2d.Y()
        self.width = bb2d.Width()
        self.height = bb2d.Height()

    def build_tlsize(self, xpos, ypos, width, height):
        self.xpos = xpos
        self.ypos = ypos
        self.width = width
        self.height = height

    def build_minmax(self, xmin, ymin, xmax, ymax):
        self.xpos = xmin
        self.ypos = ymin
        self.height = ymax - ymin
        self.width = xmax - xmin

    def is_sane(self):
        if self.width < 1:
            return False

        if self.height < 1:
            return False

        return True

    def is_in_image(self, image_sz):
        if self.xpos + self.width < 0:
            return False

        if self.ypos + self.height < 0:
            return False

        if self.xpos > image_sz[1]:
            return False

        if self.ypos > image_sz[0]:
            return False

        return True

    def area(self):
        return self.width * self.height

    def __str__(self):
        return "x : {}, y : {}, w : {}, h : {}".format(
            self.xpos, self.ypos, self.width, self.height
        )

    def __eq__(self, other):
        return (
            self.xpos == other.xpos
            and self.ypos == other.ypos
            and self.width == other.width
            and self.height == other.height
        )

    def otb_string(self):
        return "{}, {}, {}, {}".format(self.xpos, self.ypos, self.width, self.height)


class SingleTargetTrackSequence:
    def __init__(self, data_basepath, sequence_name, num_frames, bb_type="tl-size"):
        self.data_basepath = Path(data_basepath)
        self.gt_basepath = Path(data_basepath) / "anno"
        self.sequence_path = self.data_basepath / sequence_name
        self.gt_filename = self.gt_basepath / "{}.txt".format(sequence_name)
        self.images_dir = self.sequence_path / "img"
        self.sequence_name = sequence_name
        self.num_frames = num_frames
        self.bb_type = bb_type

        with open(self.gt_filename) as gt_file:
            bb_text = gt_file.read().splitlines()

        self.bounding_boxes = []
        for line in bb_text:
            line_parts = [int(part) for part in line.replace("\t", ",").split(",")]
            bbox = BoundingBox(self.bb_type, *line_parts)
            self.bounding_boxes.append(bbox)

        self.sequence_filenames = sorted(
            [Path(filename) for filename in os.listdir(self.images_dir)]
        )

    def list_frames(self):
        for frame in self.sequence_filenames:
            print(frame)

    def check_frames(self):
        for filename_idx, filename in enumerate(self.sequence_filenames):
            assert os.path.exists(
                self.images_dir / filename
            ), "Could not find frame {}, was number: {}".format(filename, filename_idx)

    def __getitem__(self, frame_idx):
        filename = self.sequence_filenames[frame_idx]
        full_path = self.images_dir / filename
        print(full_path)
        image = read_image(full_path)
        bounding_box = self.bounding_boxes[frame_idx]

        return {"image": image, "bounding_box": bounding_box}

    def __len__(self):
        return len(self.sequence_filenames)


class OnlineTrackingBenchmark:
    def __init__(self, idx, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.sequences = [
            SingleTargetTrackSequence(self.dataset_path, "DragonBaby", num_frames=113),
            SingleTargetTrackSequence(
                self.dataset_path, "DragonBaby_static", num_frames=5
            ),
        ]
        # self.sequences = [
        #     SingleTargetTrackSequence(self.dataset_path, "Basketball", num_frames=725),
        #     SingleTargetTrackSequence(self.dataset_path, "Biker", num_frames=142),
        #     SingleTargetTrackSequence(self.dataset_path, "BlurBody", num_frames=334),
        #     SingleTargetTrackSequence(self.dataset_path, "BlurCar3", num_frames=359),
        #     SingleTargetTrackSequence(self.dataset_path, "Bolt", num_frames=350),
        #     SingleTargetTrackSequence(self.dataset_path, "Box", num_frames=670),
        #     SingleTargetTrackSequence(self.dataset_path, "CarScale", num_frames=252),
        #     SingleTargetTrackSequence(self.dataset_path, "Coke", num_frames=291),
        #     SingleTargetTrackSequence(self.dataset_path, "Coupon", num_frames=327),
        #     SingleTargetTrackSequence(self.dataset_path, "Crossing", num_frames=120),
        #     SingleTargetTrackSequence(self.dataset_path, "Crowds", num_frames=347),
        #     SingleTargetTrackSequence(self.dataset_path, "DragonBaby", num_frames=113),
        #     SingleTargetTrackSequence(self.dataset_path, "FaceOcc1", num_frames=113),
        #     SingleTargetTrackSequence(self.dataset_path, "Human7", num_frames=250),
        #     SingleTargetTrackSequence(self.dataset_path, "Human9", num_frames=250),
        #     SingleTargetTrackSequence(self.dataset_path, "Ironman", num_frames=166),
        #     SingleTargetTrackSequence(self.dataset_path, "Jogging", num_frames=307),
        #     SingleTargetTrackSequence(self.dataset_path, "KiteSurf", num_frames=84),
        #     SingleTargetTrackSequence(self.dataset_path, "Liquor", num_frames=1200),
        #     SingleTargetTrackSequence(self.dataset_path, "Man", num_frames=134),
        #     SingleTargetTrackSequence(
        #         self.dataset_path, "MotorRolling", num_frames=164
        #     ),
        #     SingleTargetTrackSequence(self.dataset_path, "Shaking", num_frames=365),
        #     SingleTargetTrackSequence(self.dataset_path, "Singer2", num_frames=366),
        #     SingleTargetTrackSequence(self.dataset_path, "Soccer", num_frames=392),
        #     SingleTargetTrackSequence(self.dataset_path, "Subway", num_frames=175),
        #     SingleTargetTrackSequence(self.dataset_path, "Surfer", num_frames=376),
        #     SingleTargetTrackSequence(self.dataset_path, "Tiger1", num_frames=354),
        #     SingleTargetTrackSequence(self.dataset_path, "Trans", num_frames=124),
        #     SingleTargetTrackSequence(self.dataset_path, "Walking2", num_frames=500),
        #     SingleTargetTrackSequence(self.dataset_path, "Walking", num_frames=412),
        # ]
        self.idx = idx

    def __getitem__(self, idx):
        return self.sequences[idx]

    def __len__(self):
        return len(self.sequences)

    def list_sequences(self):
        for seq_idx, seq_name in enumerate(self.sequence_names):
            print("{}: {}".format(seq_idx, seq_name))

    def calculate_per_frame_iou(self, sequence_idx, tracked_boxes):
        iou = []
        for frame_idx, frame_data in enumerate(self.sequences[self.idx[sequence_idx]]):
            if frame_idx == 0:
                continue
            gt_box = frame_data["bounding_box"]
            try:
                union = gt_box.union_box(tracked_boxes[frame_idx])
                intersection = gt_box.intersection_box(tracked_boxes[frame_idx])
                iou.append(intersection.area() / union.area())
            except IndexError:
                iou.append(0)

        return iou

    def calculate_auc(self, sequence_idx, tracked_boxes):
        seq_name = self.sequences[sequence_idx].sequence_name
        per_frame_iou = self.calculate_per_frame_iou(sequence_idx, tracked_boxes)
        # auc = np.cumsum(per_frame_iou)
        print(per_frame_iou)
        auc = sum(per_frame_iou) / len(per_frame_iou)
        return auc

    def calculate_performance(self, per_sequence_performance):
        per_seq_auc = []
        for sequence_idx, seq_tracker_output in enumerate(per_sequence_performance):
            per_seq_auc.append(self.calculate_auc(sequence_idx, seq_tracker_output))

        return per_seq_auc
