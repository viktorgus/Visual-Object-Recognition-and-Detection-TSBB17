# Implicit that this is a test

from cvl.dataset import OnlineTrackingBenchmark

otb = OnlineTrackingBenchmark("/media/gusha40/smaugsung/TSBB17/otb_mini")

# otb[16].list_frames()
# otb[16].check_frames()

for seq_idx, seq in enumerate(otb):
    print("Sequence: {}, {}".format(seq_idx, seq.sequence_name))
    for frame_idx, frame in enumerate(otb[seq_idx]):
        print("  {}".format(frame_idx))
