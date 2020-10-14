import json

import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional_tensor import _rgb2hsv

from trainer import Trainer
from tester import Tester
from models.cvlNet import cvlNet
from models.goalNet_diffPoolSizes import goalNet
#from models.goalNet_spatial_pooling import goalNet
#from models.goalNet_average_max_pool import goalNet
#from models.goalNet import goalNet

from models.multiStreamNet import multiStream


class CombineRGBHSV(object):
    def __call__(self, sample):
        return {"rgb": sample, "hsv": _rgb2hsv(sample)}


def load_data(args):
    # Image transformations to apply to all images in the dataset (Data Augmentation)
    transform = transforms.Compose(
        [
            #transforms.RandomResizedCrop((30,30), scale=(0.08,1)),           
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
           # CombineRGBHSV(),
        ]
    )

    train_loader = None

    if args["do_train"]:
        # Specify the path to the CIFAR-10 dataset and create a dataloader
        train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args["train_batch_size"], shuffle=True, num_workers=2
        )

    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args["test_batch_size"], shuffle=False, num_workers=2
    )

    return train_loader, test_loader


def save_results(args, metrics):
    output = {"args": args, "metrics": metrics}

    with open("results/" + args["model_name"] + ".json", "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    torch.manual_seed(0)

    args = {
        "use_cuda": True,
        "train_batch_size": 256,
        "test_batch_size": 128,
        "lr": 0.001,
        "gamma": 0.3,
        "start_epoch": 1,
        "num_epochs": 50,
        "init_type": "None",  # None, "uniform", "normal", "xavier"
        "pooling_type": "max",
        "dropout_type": "spatial",  # "spatial", "channel"
        "dropout_p": 0.5,
        "patience": 10,
        "tolerance": -0.0001,
        "do_train": True,
        "load_model": False,
        "model_name": "lr_test",
        "save_checkpoints": True,
        "save_model": True,
    }

    # Load and initialize the network architecture
    model = goalNet(args)

    if args["load_model"]:
        model.load_state_dict(torch.load(f"model_files/{args['model_name']}/final.pt"))

    train_loader, test_loader = load_data(args)

    if args["use_cuda"]:
        model.cuda()
        cudnn.benchmark = True

    if args["do_train"]:
        trainer = Trainer(args, model)
        metrics = trainer.train(args, train_loader, test_loader)
        save_results(args, metrics)
    else:
        tester = Tester(args)
        test_acc = tester.test(model, test_loader)
        print(test_acc)
