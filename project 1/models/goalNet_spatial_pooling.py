import torch.nn as nn
import torch.nn.functional as F
import torch


class goalNet(nn.Module):
    def __init__(self, args):

        super(goalNet, self).__init__()
        
        p = args["dropout_p"]
        dropoutStringDict = {"spatial" : nn.Dropout(p), "channel" : nn.Dropout2d(p)}
        self.dropout_type = args["dropout_type"]
        self.dropout = None
        if self.dropout_type in dropoutStringDict : self.dropout = dropoutStringDict[self.dropout_type]

        self.pooling_type = args["pooling_type"]
        poolingStringDict = {"max" : nn.MaxPool2d, "average" : nn.AvgPool2d }
        self.pooling = poolingStringDict[self.pooling_type] 


        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool1 = self.pooling(4, 4)

        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.pool2 = self.pooling(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool3 = self.pooling(2, 2)

        self.conv4 = nn.Conv2d(64, 64, 4, padding=2)
        self.prediction = nn.Linear(1344, 10)
        self.loss = nn.LogSoftmax(1)

        self.spatial1 = nn.AdaptiveMaxPool2d(1)
        self.spatial2 = nn.AdaptiveMaxPool2d(2)
        self.spatial3 = nn.AdaptiveMaxPool2d(4)



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
        out1 = self.spatial1(out)
        out2 = self.spatial2(out)
        out3 = self.spatial3(out)
        

        out1 = out1.view(-1, 64) 
        out2 = out2.view(-1, 256)
        out3 = out3.view(-1, 1024)

        out = torch.cat((out1,out2,out3),1)
        
        if self.dropout and self.dropout_type == "spatial" : out = self.dropout(out)
        out = self.prediction(out)
        
        out = self.loss(out)

        return out
