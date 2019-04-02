import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models import resnet
from torch.nn import functional as F
from scse import SCSEBlock
from torchsummary import summary

class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)

        return x


class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.SCSEBlockARM = SCSEBlock(128)

    def forward(self, input):
        print('Input', input.shape)
        # global average pooling
        x = self.avgpool(input)
        print(x.shape)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        print('after conv', x.shape)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        print(x.shape)
        print('\n')
        return x

class LinkNet(nn.Module):
    """
    Generate Model Architecture
    """

    def __init__(self, n_classes=21):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        super(LinkNet, self).__init__()

        base = resnet.resnet18(pretrained=True)

        self.in_block = nn.Sequential(
            base.conv1,
            base.bn1,
            base.relu,
            base.maxpool
        )

        self.encoder1 = base.layer1
        self.encoder2 = base.layer2
        self.encoder3 = base.layer3
        self.encoder4 = base.layer4

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, 2, 1, 1),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 2, 2, 0)
        #self.lsm = nn.LogSoftmax(dim=1)
        self.lsm = nn.Sigmoid()
        self.SCSEBlock4 = SCSEBlock(256)
        self.SCSEBlock3 = SCSEBlock(128)
        self.SCSEBlock2 = SCSEBlock(64)
        self.SCSEBlock1 = SCSEBlock(64)
        self.ARM1 = AttentionRefinementModule(64, 64)
        self.ARM2 = AttentionRefinementModule(64, 64)
        self.ARM3 = AttentionRefinementModule(128, 128)
        self.ARM4 = AttentionRefinementModule(256, 256)


    def forward(self, x):
        # Initial block
        #print(x.size())
        x = self.in_block(x)

        # Encoder blocks
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.decoder4(e4) + self.ARM4(e3)
        d4 = self.SCSEBlock4(d4)
        d3 = self.decoder3(d4) + self.ARM3(e2)
        d3 = self.SCSEBlock3(d3)
        d2 = self.decoder2(d3) + self.ARM2(e1)
        d2 = self.SCSEBlock2(d2)
        d1 = self.decoder1(d2) + self.ARM1(x)
        d1 = self.SCSEBlock1(d1)

        # Classifier
        y = self.tp_conv1(d1)
        y = self.conv2(y)
        y = self.tp_conv2(y)
        y = self.lsm(y)

        return y
    
if __name__== '__main__':
    model= LinkNet(n_classes=1).cuda()
    #summary(model, input_size=(3,1440,1440))
    x = torch.Tensor(1,3,1440,1440).cuda()
    out = model(x)