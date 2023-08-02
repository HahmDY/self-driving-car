import torch
from torch import nn

class CustomNetwork(nn.Module):
    def __init__(self):
        super(CustomNetwork, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=2,
                                stride=1, padding=0, padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=3, out_channels=24, kernel_size=2,
                               stride=1, padding=0, padding_mode='zeros')

        self.layer1 = nn.Linear(72, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

        self.activation = nn.Tanh()

        pass

    def forward(self, input):
        x=input.reshape(1,1,5)
        x = self.conv1(x)
        x = self.conv2(x)
        x = torch.flatten(x, 1)
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        out = self.layer3(x)
        return out