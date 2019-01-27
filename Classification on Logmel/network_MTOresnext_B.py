
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch
import pretrainedmodels


class WaveResnext(nn.Module):
    def __init__(self, modules, num_classes):
        super(WaveResnext, self).__init__()
        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=11, stride=1, padding=5)
        self.conv1_2 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=51, stride=5, padding=25)
        self.conv1_3 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=101, stride=10, padding=50)

        self.bn1_1 = nn.BatchNorm1d(32)
        self.bn1_2 = nn.BatchNorm1d(32)
        self.bn1_3 = nn.BatchNorm1d(32)

        self.conv2_1 = nn.Conv1d(in_channels=32, out_channels=220, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=32, out_channels=220, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv1d(in_channels=32, out_channels=220, kernel_size=3, stride=1, padding=1)

        self.bn2_1 = nn.BatchNorm1d(220)
        self.bn2_2 = nn.BatchNorm1d(220)
        self.bn2_3 = nn.BatchNorm1d(220)

        self.pool2_1 = nn.MaxPool1d(kernel_size=150, stride=150)
        self.pool2_2 = nn.MaxPool1d(kernel_size=30, stride=30)
        self.pool2_3 = nn.MaxPool1d(kernel_size=15, stride=15)

        self.resBlocks = nn.Sequential(*modules)

        self.avgpool = nn.AvgPool2d((7, 7), stride=(1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.bn1_1(self.conv1_1(x)))
        x2 = self.relu(self.bn1_2(self.conv1_2(x)))
        x3 = self.relu(self.bn1_3(self.conv1_3(x)))

        x1 = self.relu(self.bn2_1(self.conv2_1(x1)))
        x2 = self.relu(self.bn2_2(self.conv2_2(x2)))
        x3 = self.relu(self.bn2_3(self.conv2_3(x3)))

        x1 = self.pool2_1(x1)
        x2 = self.pool2_2(x2)
        x3 = self.pool2_3(x3)  # (batchSize, 32L, 441L)

        x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        x3 = torch.unsqueeze(x3, 1)  # (batchSize, 1L, 32L, 441L)

        # x = torch.cat((x1, x2, x3), dim=2) #(batchSize, 1L, 96L, 441L)
        x = torch.cat((x1, x2, x3), dim=1)  # (batchSize, 3L, 96L, 441L)
#         print("0:", x.size())
        # x = self.conv0(x)
        # x = self.bn0(x)
        # x = self.relu(x)
        # x = self.maxpool(x)
        x = self.resBlocks(x)
#         print("1:", x.size())
        x = self.avgpool(x)
#         print("2:", x.size())
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def waveResnext101_32x4d(pretrained='imagenet', num_classes=16):

    base = pretrainedmodels.resnext101_32x4d(pretrained=pretrained)
    base.avg_pool = nn.AvgPool2d((2, 5), stride=(2, 5))
    base.last_linear = nn.Linear(512 * 4, 16)
    modules = list(base.children())
    # print(type(modules))
    # print(len(modules))
    model = WaveResnext(modules[0], num_classes)

    return model

if __name__ == '__main__':
    model = waveResnext101_32x4d(pretrained='imagenet')

