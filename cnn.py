import torch.nn as nn

class CNN:

    def __init__(self, device):
        self.device = device
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(15, 15)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(p=0.2),
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(10, 10), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Dropout(p=0.2),
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(6272, 512),
            nn.ReLU(),
            nn.Linear(512, 2),
            nn.Softmax(),
        ).to(self.device)

    def forward(self, input):
        return self.net(input)

    # def buildNet(self):
    #     for l in range(self.layers):
    #         self.network[l] = nn.Conv2d
