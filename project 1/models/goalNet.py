import torch.nn as nn
import torch.nn.functional as F


class goalNet(nn.Module):
    def __init__(self, args):

        super(goalNet, self).__init__()

        p = args["dropout_p"]
        dropout_dict = {"spatial": nn.Dropout(p), "channel": nn.Dropout2d(p)}
        pooling_dict = {"max": nn.MaxPool2d, "average": nn.AvgPool2d}
        init_dict = {
            "uniform": nn.init.uniform_,
            "normal": nn.init.normal_,
            "xavier": nn.init.xavier_normal_,
        }

        self.dropout_type = args["dropout_type"]
        self.dropout = None
        if self.dropout_type in dropout_dict:
            self.dropout = dropout_dict[self.dropout_type]
        
        self.pooling_type = args["pooling_type"]
        self.pooling = pooling_dict[self.pooling_type]

        self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
        self.pool1 = self.pooling(4, 4)

        self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
        self.pool2 = self.pooling(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
        self.pool3 = self.pooling(2, 2)

        self.conv4 = nn.Conv2d(64, 64, 4, padding=2)
        self.prediction = nn.Linear(576, 10)
        self.loss = nn.LogSoftmax(1)

        if args["init_type"] in init_dict:
            self.init = init_dict[args["init_type"]]

            for module in self.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if isinstance(module, nn.Conv2d):
                        n = module.in_channels
                        for k in module.kernel_size:
                            n *= k
                    else:
                        n = module.weight.size(1)

                    stdv = 1. / math.sqrt(n)
                    print(stdv)
                    self.init(module.weight, -stdv, stdv)

                    if args["init_type"] != "xavier":
                        self.init(module.bias, -stdv, stdv)
                    else:
                        nn.init.zeros_(module.bias, -stdv, stdv)

    def forward(self, x):

        out = self.conv1(x)
        if self.dropout and self.dropout_type == "channel":
            out = self.dropout(out)
        out = self.pool1(out)
        out = F.relu(out)

        out = self.conv2(out)
        if self.dropout and self.dropout_type == "channel":
            out = self.dropout(out)
        out = F.relu(out)
        out = self.pool2(out)

        out = self.conv3(out)
        if self.dropout and self.dropout_type == "channel":
            out = self.dropout(out)
        out = F.relu(out)
        out = self.pool3(out)

        out = self.conv4(out)
        if self.dropout and self.dropout_type == "channel":
            out = self.dropout(out)
        out = F.relu(out)
        
        out = out.view(-1, 576)

        if self.dropout and self.dropout_type == "spatial":
            out = self.dropout(out)
        out = self.prediction(out)

        out = self.loss(out)

        return out
