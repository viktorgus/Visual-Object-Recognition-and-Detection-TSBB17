# Visual Object Recognition and Detection TSBB17
Labs using PyTorch and openCV for object recognition and generalised object tracking.

# Lab 1
Implemented and tested various setups for a CNN for image recognition. Experimented with different pooling settings, dropout, multi-stream networks, spatial pyramid pooling, different weight initializations, and hyperparameter tuning such as learning rate. 

Result: achieved over 90% test accuracy on the CIFAR10 dataset.

# Lab 2
Implemented generalized object trackers from the ground up using DCF with convulutions using dot product in the fourier space.

- Implemented MOSSE tracker in cvl/trackers/mosse_tracker.py
- Implemented a multi-channel MOSSE in cvl/trackers/mosse_multi.py
- Implemented a scale estimation component in MOSSE in cvl/trackers/mosse_multi_scale.py
- Implemented several new features and experimentation with different parameter settings for theese: Histogram of gradients, gradient filter bank, deep features (features from output of different layers in pretrained alexNet)