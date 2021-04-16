import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) * 2 """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x
    
class ConvBlock(nn.Module):
    """ (2D conv => BN => LeakyReLU) """

    def __init__(self, in_ch, out_ch, k_size, pad, dil):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k_size, padding=pad, dilation=dil),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block(x)
        return x

    
class MVNet(nn.Module):
    """
    Multi-View Network (MV-Net)

    PARAMETERS
    ----------
    n_classes: int
        Number of classes used for the semantic segmentation task
    n_frames: int
        Total numer of frames used as a sequence
    """
    def __init__(self, n_classes, n_frames):
        super().__init__()
        self.n_classes = n_classes
        self.n_frames = n_frames

        # Encoding
        self.rd_double_conv_block1 = DoubleConvBlock(in_ch=self.n_frames, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block1 = DoubleConvBlock(in_ch=self.n_frames, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.rd_max_pool = nn.MaxPool2d(2, stride=(2, 1))
        self.max_pool = nn.MaxPool2d(2, stride=2)
        self.rd_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block2 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.rd_single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block1_1x1 = ConvBlock(in_ch=128, out_ch=128, k_size=1, pad=0, dil=1)

        # Decoding
        self.rd_single_conv_block2_1x1 = ConvBlock(in_ch=256, out_ch=128, k_size=1, pad=0, dil=1)
        self.ra_single_conv_block2_1x1 = ConvBlock(in_ch=256, out_ch=128, k_size=1, pad=0, dil=1)
        self.rd_upconv1 = nn.ConvTranspose2d(128, 128, (2, 1), stride=(2, 1))
        self.ra_upconv1 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rd_double_conv_block3 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block3 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.rd_upconv2 = nn.ConvTranspose2d(128, 128, (2, 1), stride=(2, 1))
        self.ra_upconv2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.rd_double_conv_block4 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)
        self.ra_double_conv_block4 = DoubleConvBlock(in_ch=128, out_ch=128, k_size=3,
                                                     pad=1, dil=1)

        # Final 1D convs
        self.rd_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)
        self.ra_final = nn.Conv2d(in_channels=128, out_channels=n_classes, kernel_size=1)


    def forward(self, x_rd, x_ra):
        # Backbone
        # Preprocess rd for convolution shape matching
        x_rd = F.pad(x_rd, (0, 1, 0, 0), "constant", 0)
        x1_rd = self.rd_double_conv_block1(x_rd)
        x1_rd = self.rd_max_pool(x1_rd)
        x1_ra = self.rd_double_conv_block1(x_ra)
        x1_ra = self.max_pool(x1_ra)

        x1_rd = F.pad(x1_rd, (0, 1, 0, 0), "constant", 0)
        x2_rd = self.rd_double_conv_block2(x1_rd)
        x2_rd = self.rd_max_pool(x2_rd)
        x2_ra = self.rd_double_conv_block2(x1_ra)
        x2_ra = self.max_pool(x2_ra)

        x3_rd = self.rd_single_conv_block1_1x1(x2_rd)
        x3_ra = self.ra_single_conv_block1_1x1(x2_ra)

        # Latent Space
        x4 = torch.cat((x3_rd, x3_ra), 1)

        # Decoding with upconvs
        x5_rd = self.rd_single_conv_block2_1x1(x4)
        x5_ra = self.ra_single_conv_block2_1x1(x4)

        x6_rd = self.rd_upconv1(x5_rd)
        x6_ra = self.ra_upconv1(x5_ra)
        x7_rd = self.rd_double_conv_block3(x6_rd)
        x7_ra = self.ra_double_conv_block3(x6_ra)

        x8_rd = self.rd_upconv2(x7_rd)
        x8_ra = self.ra_upconv2(x7_ra)
        x9_rd = self.rd_double_conv_block3(x8_rd)
        x9_ra = self.ra_double_conv_block3(x8_ra)

        # Final 1D convolutions
        x10_rd = self.rd_final(x9_rd)
        x10_ra = self.ra_final(x9_ra)

        return x10_rd, x10_ra
