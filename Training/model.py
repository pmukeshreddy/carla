import torch
import torch.nn as nn

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = torch.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1, use_bn=True):
        super().__init__()
        layers = []
        
        # First conv block
        layers.append(nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Second conv block
        layers.append(nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Dropout for regularization
        if dropout is not None and dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        
        self.double_conv = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.double_conv(x)


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3, dropout=0.1):
        super().__init__()
        
        # Encoder – 4 levels, with BN + dropout
        self.enc1 = DoubleConv(in_channels, 64,  dropout=dropout)
        self.enc2 = DoubleConv(64, 128,        dropout=dropout)
        self.enc3 = DoubleConv(128, 256,       dropout=dropout)
        self.enc4 = DoubleConv(256, 512,       dropout=dropout)
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024, dropout=dropout)
        
        # Attention Gates
        self.att4 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.att3 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.att2 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.att1 = AttentionGate(F_g=64,  F_l=64,  F_int=32)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4   = DoubleConv(1024, 512, dropout=dropout)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3   = DoubleConv(512, 256, dropout=dropout)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2   = DoubleConv(256, 128, dropout=dropout)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1   = DoubleConv(128, 64, dropout=dropout)
        
        # Output
        self.out = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # Bottleneck
        b = self.bottleneck(self.pool(e4))
        
        # Decoder with Attention
        d4 = self.upconv4(b)
        e4_att = self.att4(g=d4, x=e4)
        d4 = torch.cat([d4, e4_att], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.upconv3(d4)
        e3_att = self.att3(g=d3, x=e3)
        d3 = torch.cat([d3, e3_att], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        e2_att = self.att2(g=d2, x=e2)
        d2 = torch.cat([d2, e2_att], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        e1_att = self.att1(g=d1, x=e1)
        d1 = torch.cat([d1, e1_att], dim=1)
        d1 = self.dec1(d1)
        
        return self.out(d1)
