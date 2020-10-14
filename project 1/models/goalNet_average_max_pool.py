import torch.nn as nn
import torch.nn.functional as F


class goalNet(nn.Module):
    def __init__(self, args):

        super(goalNet, self).__init__()
        
        p = args["dropout_p"]
        dropoutStringDict = {"spatial" : nn.Dropout(p), "channel" : nn.Dropout2d(p)}
        self.dropout_type = args["dropout_type"]
        self.dropout = None
        if self.dropout_type in dropoutStringDict : self.dropout = dropoutStringDict[self.dropout_type]

        self.pooling1 = nn.AvgPool2d
        self.pooling2 = nn.MaxPool2d 


        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool1 = self.pooling1(4, 4)

        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.pool2 = self.pooling1(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool3 = self.pooling2(2, 2)

        self.conv4 = nn.Conv2d(64, 64, 4, padding=2)
        self.prediction = nn.Linear(576, 10)
        self.loss = nn.LogSoftmax(1)


    def forward(self, x):

        out = self.conv1(x)
        if self.dropout and self.dropout_type == "channel" : out = self.dropout(out)
        out = self.pool1(out)
        out = F.relu(out)
       

        out = self.conv2(out)
        if self.dropout and self.dropout_type == "channel" : out = self.dropout(out)
        out = F.relu(out)
        out = self.pool2(out)
       

        out = self.conv3(out)
        if self.dropout and self.dropout_type == "channel" : out = self.dropout(out)
        out = F.relu(out)
        out = self.pool3(out)

        out = self.conv4(out)
        if self.dropout and self.dropout_type == "channel" : out = self.dropout(out)
        out = F.relu(out)

        out = out.view(-1, 576)

        if self.dropout and self.dropout_type == "spatial" : out = self.dropout(out)
        out = self.prediction(out)
        
        out = self.loss(out)

        return out
