import os
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import numpy as np

from tqdm import tqdm

from tester import Tester


class Trainer:
    def __init__(self, args, model):
        self.model = model

        # The objective (loss) function
        self.objective = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=args["lr"])
        self.scheduler = StepLR(self.optimizer, step_size=10, gamma=args["gamma"])

        self.tester = Tester(args)

        if not os.path.isdir("model_files"):
            os.mkdir("model_files")
        self.base_path = f"model_files/{args['model_name']}"
        if not os.path.isdir(self.base_path):
            os.mkdir(self.base_path)

    def save_checkpoint(self, epoch):
        """Save a checkpoint when the epoch finishes"""
        state = {
            "epoch": epoch,
            "net": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }

        file_path = f"{self.base_path}/epoch_{epoch}.ckpt"
        torch.save(state, file_path)

    def train(self, args, train_loader, test_loader=None):
        train_loss, train_acc, test_acc, roc_auc, test_loss = [], [], [], [], []
        start_time = time.time()

        for epoch in tqdm(
            range(args["start_epoch"], args["num_epochs"] + 1), desc="Epoch"
        ):
            self.model.train()

            # Variables for training status
            temp_loss, correct, total = 0, 0, 0

            # Loop on all images in the dataset
            for batch_idx, (inputs, targets) in enumerate(
                tqdm(train_loader, desc="Iteration")
            ):

                # Move data to GPU if CUDA is available
                
                if isinstance(inputs, dict):
                    inputs_rgb = torch.tensor(inputs["rgb"], requires_grad=True)
                    inputs_hsv = torch.tensor(inputs["hsv"], requires_grad=True)
                    if args["use_cuda"]:
                        inputs_rgb, inputs_hsv = inputs_rgb.cuda(), inputs_hsv.cuda()
                    inputs = (inputs_rgb, inputs_hsv)
                else:
                    inputs = torch.tensor(inputs, requires_grad=True)
                    inputs = inputs.cuda()

                if args["use_cuda"]:
                    targets = targets.cuda()
                targets = torch.tensor(targets).long()

                self.optimizer.zero_grad()  # Clear the gradients of all parameters
                outputs = self.model(inputs)  # Forward pass
                loss = self.objective(outputs, targets)  # Calculate loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update model parameters

                # Update training status
                temp_loss += loss.item()

                # Find the class with the highest output
                _, predicted = torch.max(outputs.data, 1)

                # Count number total number of images trained so far and the
                # correctly classified ones
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            train_loss += [temp_loss / (batch_idx + 1)]
            train_acc += [100.0 * correct / total]

            if args["save_checkpoints"]:
                self.save_checkpoint(epoch)

            if test_loader:
                test_metrics = self.tester.test(self.model, test_loader)
                test_acc += [test_metrics[0]]
                roc_auc += [test_metrics[1]]
                test_loss += test_metrics[2]

            self.scheduler.step()  # Update learning rate schedule

            # Checking early stopping criterias
            x = []
            y = []

            if epoch > args["patience"]:
                for i in range(args["patience"]):
                    x += [epoch - 1 - i]
                    y += [test_loss[epoch - 1 - i]]

                poly = np.polyfit(np.array(x),np.array(y), 1)
                print(poly)
                if poly[0] > args["tolerance"]:
                        break

        train_time = time.time() - start_time

        if args["save_model"]:
            file_path = f"{self.base_path}/final.pt"
            torch.save(self.model.state_dict(), file_path)

        print(test_acc[-1])
        metrics = {
            "train_time": train_time,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "roc_auc": roc_auc,
        }

        return metrics
