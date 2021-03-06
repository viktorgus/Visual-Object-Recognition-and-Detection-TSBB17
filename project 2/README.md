# TSBB17 Project 2: Visual Object Tracking
Lukas Borggren, Viktor Gustafsson and Gustav Wahlquist.  
See the /cvl folder for models used.

## Final Performance
![tracker_performance](https://user-images.githubusercontent.com/46990011/107772698-dbeade00-6d3c-11eb-897e-bff9c2fa96c2.png)

## Introduction
This project aims to evaluate the MOSSE tracker and how it performs with different maps of features on
the OTB-mini dataset.

# Dataset
The dataset used in this project is a subset of the OTB benchmark called OTB-mini. It consists of 30
image sequences together with a ground truth bounding box for each frame. The sequences include a
varying number of frames, but the size of the frames are consistent across the whole dataset.
During evaluation, the learning rate and regularization term are both fixed to 0.01 for the MOSSE tracker.
For the two-dimensional Gaussian function, 𝜇 = 0 and 𝜎 = 2 is used. These parameters were varied and
tested for a few iterations on the MOSSE tracker with different feature configurations before settling on
the final values.


## Installation
To create an environment where the code can be run, run the following:
```
git clone https://gitlab.liu.se/lukbo262/tsbb17-project2
cd tsbb17-project2
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
