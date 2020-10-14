import torch.nn as nn
import torch.nn.functional as F


class cvlNet(nn.Module):
    def __init__(self):

        super(cvlNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.fc1 = nn.Conv2d(
            32, 32, 4
        )  # Fully connected since expects a 4x4xc feature map
        self.prediction = nn.Linear(32, 10)
        self.loss = nn.LogSoftmax(1)

    def forward(self, x):
        out = self.conv1(x)
        out = F.max_pool2d(out, kernel_size=(8, 8), stride=8)  # outputs 4x4xc featurmap
        out = F.relu(out)  # relu1

        out = self.fc1(out)
        out = F.relu(out)  # relu2

        out = out.view(-1, 32)

        out = self.prediction(out)

        out = self.loss(out)

        return out
