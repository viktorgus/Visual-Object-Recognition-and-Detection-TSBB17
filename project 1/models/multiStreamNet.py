import torch
import torch.nn as nn
import torch.nn.functional as F


class singleStream(nn.Module):
    def __init__(self, args):
        super(singleStream, self).__init__()

        p = args["dropout_p"]
        dropout_dict = {"spatial": nn.Dropout(p), "channel": nn.Dropout2d(p)}
        pooling_dict = {"max": nn.MaxPool2d, "average": nn.AvgPool2d}

        self.dropout_type = args["dropout_type"]
        self.dropout = None
        if self.dropout_type in dropout_dict:
            self.dropout = dropout_dict[self.dropout_type]

        self.pooling_type = args["pooling_type"]
        self.pooling = pooling_dict[self.pooling_type]

        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool1 = self.pooling(4, 4)

    def forward(self, out):
        out = self.conv1(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.pool1(out)
        out = F.relu(out)

        return out


class multiStream(nn.Module):
    def __init__(self, args):
        super(multiStream, self).__init__()

        self.rgb_stream = singleStream(args)
        self.hsv_stream = singleStream(args)

        p = args["dropout_p"]
        dropout_dict = {"spatial": nn.Dropout(p), "channel": nn.Dropout2d(p)}
        pooling_dict = {"max": nn.MaxPool2d, "average": nn.AvgPool2d}

        self.pooling_type = args["pooling_type"]
        self.pooling = pooling_dict[self.pooling_type]

        self.dropout_type = args["dropout_type"]
        self.dropout = None
        if self.dropout_type in dropout_dict:
            self.dropout = dropout_dict[self.dropout_type]

        self.conv1 = nn.Conv2d(6, 32, 5, padding=2)
        self.pool1 = self.pooling(4, 4)

        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.pool2 = self.pooling(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool3 = self.pooling(2, 2)

        self.conv4 = nn.Conv2d(64, 64, 4, padding=2)
        self.prediction = nn.Linear(576, 10)
        self.loss = nn.LogSoftmax(1)

    def forward(self, x):

        out = torch.cat((x[0], x[1]), dim=1)

        out = self.conv1(out)
        if self.dropout:
            out = self.dropout(out)
        out = self.pool1(out)
        out = F.relu(out)

        out = self.conv2(out)
        if self.dropout:
            out = self.dropout(out)
        out = F.relu(out)
        out = self.pool2(out)

        out = self.conv3(out)
        if self.dropout:
            out = self.dropout(out)
        out = F.relu(out)
        out = self.pool3(out)

        out = self.conv4(out)
        if self.dropout:
            out = self.dropout(out)
        out = F.relu(out)

        if self.dropout:
            out = self.dropout(out)

        out = out.view(-1, 576)
        out = self.prediction(out)
        out = self.loss(out)

        return out
