import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import OneHotEncoder


class Tester:
    def __init__(self, args):

        self.objective = nn.CrossEntropyLoss()
        self.use_cuda = args["use_cuda"]

        # Specify class labels
        self.classes = (
            "plane",
            "car",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        )

    def test(self, model, test_loader):
        test_loss, all_pred, all_target = [], [], []
        temp_loss, correct, total = 0, 0, 0
        model.eval()

        for idx, (inputs, targets) in enumerate(test_loader):
            # Move data to GPU if CUDA is available
                
            if isinstance(inputs, dict):
                inputs_rgb = torch.tensor(inputs["rgb"], requires_grad=True)
                inputs_hsv = torch.tensor(inputs["hsv"], requires_grad=True)
                if self.use_cuda:
                    inputs_rgb, inputs_hsv = inputs_rgb.cuda(), inputs_hsv.cuda()
                inputs = (inputs_rgb, inputs_hsv)
            else:
                inputs = torch.tensor(inputs, requires_grad=True)
                inputs = inputs.cuda()

            if self.use_cuda:
                targets = targets.cuda()

            outputs = model(inputs)

            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            all_pred += predicted.tolist()
            all_target += targets.tolist()

            temp_loss += self.objective(outputs, targets).item()  # Calculate loss

        num_classes = len(self.classes)
        num_pred = len(all_pred)

        oh_pred = np.zeros((num_pred, num_classes))
        for ind, class_ind in enumerate(all_pred):
            oh_pred[ind][class_ind] = 1

        oh_target = np.zeros((num_pred, num_classes))
        for ind, class_ind in enumerate(all_target):
            oh_target[ind][class_ind] = 1

        fpr, tpr, roc_auc = dict(), dict(), dict()

        for i in range(num_classes):
            fpr[i], tpr[i], _ = roc_curve(oh_target[:, i], oh_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        fpr["micro"], tpr["micro"], _ = roc_curve(oh_target.ravel(), oh_pred.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        test_acc = 100 * correct / total
        roc_auc = roc_auc
        test_loss += [temp_loss / (idx + 1)]


        return test_acc, roc_auc, test_loss
