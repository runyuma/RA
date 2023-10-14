import torch
import torch.nn as nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_planes, out_channels=planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(planes)
        )

        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        # print(out.shape)
        # print(self.shortcut(x).shape)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self):
        block = BasicBlock
        num_block = [2, 2, 2, 2]
        super(ResNet18, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)               # the first layer

        self.layer1 = self._make_layer(block, 64, num_block[0], stride=1)             # four layers 2-5
        self.layer2 = self._make_layer(block, 128, num_block[1], stride=2)            # four layers 6-9
        self.layer3 = self._make_layer(block, 256, num_block[2], stride=2)            # four layers 10-13
        self.layer4 = self._make_layer(block, 512, num_block[3], stride=2)            # four layers 14-17
        
        self.fc = nn.Linear(512, 10)                                # the last layer

    def _make_layer(self, block, planes, num_blocks, stride):                          
        layers = []
        for i in range(num_blocks):
            if i == 0:
                layers.append(block(self.in_planes, planes, stride))
            else:
                layers.append(block(planes, planes, 1))

        self.in_planes = planes   
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        # print("con1",x.shape)
        x = self.layer1(x)
        # print("layer1",x.shape)
        x = self.layer2(x)
        # print("layer2",x.shape)
        x = self.layer3(x)
        # print("layer3",x.shape)
        x = self.layer4(x)
        # print("layer4",x.shape)
        x = F.avg_pool2d(x,7)
        x = x.view(x.size(0), -1)
        # x = self.fc(x)
        out = x
        return out

if __name__ == "__main__":
    # test resnet
    Res = ResNet18()
    img = torch.zeros((224,224,4))
    img = img.unsqueeze(0).permute(0,3,1,2)
    img = Res(img)
    print(img.shape)

