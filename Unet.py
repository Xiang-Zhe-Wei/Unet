from torch.nn import functional as F
import torch.nn as nn
import torch

class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.CB_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.LeakyReLU()
        )
    def forward(self, x):
        return self.CB_layer(x) # model = Conv_Block(in_channels, out_channels) => model(x)


class Down_Sampling(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.DS_layer = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size=3, stride=2, padding=1, padding_mode="reflect", bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        ) 
    def forward(self, x):
        return self.DS_layer(x)


# Use interpolate rather than transpose convolution
class Up_Sampling(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.US_layer = nn.Sequential(
            nn.Conv2d(channel, channel//2, kernel_size=1, stride=1, padding=0, padding_mode="reflect", bias=False)
        ) 
    def forward(self, x, concat_feature):
        up = F.interpolate(x, scale_factor=2, mode="nearest")
        out = self.US_layer(up)
        return torch.cat((out, concat_feature), dim=1) # [N,C,H,W] concat C


class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        # go-down
        self.CB1 = Conv_Block(3, 64)        
        self.DS1 = Down_Sampling(64)        
        self.CB2 = Conv_Block(64, 128)        
        self.DS2 = Down_Sampling(128)        
        self.CB3 = Conv_Block(128, 256)        
        self.DS3 = Down_Sampling(256)        
        self.CB4 = Conv_Block(256, 512)        
        self.DS4 = Down_Sampling(512)      
        # bottom
        self.CB5 = Conv_Block(512, 1024)
        # go-up
        self.US1 = Up_Sampling(1024)
        self.CB6 = Conv_Block(1024, 512)
        self.US2 = Up_Sampling(512)
        self.CB7 = Conv_Block(512, 256)
        self.US3 = Up_Sampling(256)
        self.CB8 = Conv_Block(256, 128)
        self.US4 = Up_Sampling(128)
        self.CB9 = Conv_Block(128, 64)
        # output
        self.out = nn.Conv2d(64, out_channels=3, kernel_size=1, stride=1, padding=0)
        self.activate = nn.Sigmoid()
    def forward(self, x):
        Down1 =  self.CB1(x)
        Down2 =  self.CB2(self.DS1(Down1))
        Down3 =  self.CB3(self.DS2(Down2))
        Down4 =  self.CB4(self.DS3(Down3))
        Down5 =  self.CB5(self.DS4(Down4))
        Up1   =  self.CB6(self.US1(Down5, Down4))
        Up2   =  self.CB7(self.US2(Up1  , Down3))
        Up3   =  self.CB8(self.US3(Up2  , Down2))
        Up4   =  self.CB9(self.US4(Up3  , Down1))
        return self.activate(self.out(Up4))

if __name__ == "__main__":
    x = torch.randn(2,3,256,256)
    net = Unet()
    print(net(x).shape)

