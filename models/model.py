import torch
from torchvision import models


class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.Linear(4, 3)
        )
        self.layer2 = torch.nn.Linear(3, 6)

        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(6, 7),
            torch.nn.Linear(7, 3)
        )

        # self.conv1 = torch.nn.Conv2d(3, 6, 5)
        # self.pool = torch.nn.MaxPool2d(2, 2)
        # self.conv2 = torch.nn.Conv2d(6, 16, 5)
        # self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


alexnet = models.alexnet(weights = models.AlexNet_Weights.DEFAULT)

del alexnet.classifier

print(alexnet)


