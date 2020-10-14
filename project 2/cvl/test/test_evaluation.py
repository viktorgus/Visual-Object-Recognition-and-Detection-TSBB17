import pytest


def test_sequence_perfect():
    from cvl.dataset import OnlineTrackingBenchmark

    otb = OnlineTrackingBenchmark("/courses/TSBB17/otb_mini")
    sequence = otb[0]
    track_output = [e["bounding_box"] for e in sequence]
    print(track_output)
    error = otb.calculate_per_frame_iou(0, track_output)
    for e in error:
        assert e == 1

test_sequence_perfect()
