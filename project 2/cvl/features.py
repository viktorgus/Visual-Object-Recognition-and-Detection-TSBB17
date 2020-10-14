import torch
import torch.nn as nn

if torch.__version__ == "1.2.0":
    from torchvision.models.utils import load_state_dict_from_url
else:
    from torch.utils.model_zoo import load_url

"""
    These where taken from the torchvision repository, and modified to return the 
    features instead of the classification score.
"""

model_urls = {
    "alexnet": "http://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
}


class AlexNetFeature(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNetFeature, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, img, layers=None):
        x = torch.from_numpy(img.copy()).float()
        # Change from format H,W,C to C,H,W that pytorch expects
        x = x.permute(2,0,1)
        # Add a dimension corresponding to a batch of size 1
        x = x.unsqueeze(0)

        # If no layer specified, use all layers
        if layers==None:
            x = self.features(x)
            x = x.squeeze(0)
            x = x.permute(1,2,0).detach().numpy()
            return(x)
        
        layersToTotalLayers = {
            1:3,
            2:6,
            3:8,
            4:10,
            5:13
        }
        
        for i in range(layersToTotalLayers[layers]):
            x = self.features[i](x)
        
        x = x.squeeze(0)
        x = x.permute(1,2,0).detach().numpy()

        return x


def alexnetFeatures(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetFeature(**kwargs)
    if pretrained:
        state_dict = load_url(model_urls["alexnet"], progress=progress)
        model.load_state_dict(state_dict)
    return model
