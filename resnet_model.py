import torch
import torch.nn as nn


# For block
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1

# For first conv
GRAY_SCALE = 1  # if you want to treat RGB, change this value to 3 (didn't test however)
FIRST_OUTPUT_CHANNELS = 32
FIRST_KERNEL_SIZE = 5
FIRST_STRIDE = 2

# Global
OUTPUT_CHANNELS = 30  # 30 feature points


class _ResNetBlock(nn.Module):
    def __init__(self, in_channels, nb_filters, stride, donwsample=None, dropout=0.0):
        super(_ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=in_channels, out_channels=nb_filters,
                      kernel_size=KERNEL_SIZE, stride=stride, padding=PADDING),
            nn.BatchNorm2d(num_features=nb_filters),
            nn.ReLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(in_channels=nb_filters, out_channels=nb_filters,
                      kernel_size=KERNEL_SIZE, padding=PADDING),
        )
        self.downsample = donwsample

    def forward(self, x):
        out = self.block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return out


class ResNet(nn.Module):
    def __init__(self, dropout):
        super(ResNet, self).__init__()
        # TODO search best set of channels (and make them modifiable easily)
        self.first_conv = nn.Conv2d(in_channels=GRAY_SCALE, out_channels=FIRST_OUTPUT_CHANNELS,
                                    kernel_size=FIRST_KERNEL_SIZE, stride=FIRST_STRIDE, padding=PADDING)
        self.block_1 = self._make_block(FIRST_OUTPUT_CHANNELS, FIRST_OUTPUT_CHANNELS, STRIDE, dropout=dropout)
        self.block_2 = self._make_block(FIRST_OUTPUT_CHANNELS, FIRST_OUTPUT_CHANNELS * 2, STRIDE, dropout=dropout)
        self.block_3 = self._make_block(FIRST_OUTPUT_CHANNELS * 2, FIRST_OUTPUT_CHANNELS * 4, STRIDE, dropout=dropout)
        self.avgpool = nn.AvgPool2d(kernel_size=(6, 6))
        self.last_lin = nn.Linear(FIRST_OUTPUT_CHANNELS * 4, OUTPUT_CHANNELS)

    def _make_block(self, in_channels, out_channels, stride, dropout):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            # TODO do we really need to put batch norm 2d here ?
            downsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                          stride=stride),
            )
        return _ResNetBlock(in_channels, out_channels, stride, donwsample=downsample, dropout=dropout)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.avgpool(x)
        x = self.last_lin(x.reshape(x.size(0), -1))  # reshape (Batch, Channels, 1, 1) to (Batch, Channels)
        return x


# for debug purpose
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    x = torch.rand(32, 1, 96, 96).to(device)
    print(x.size())
    model = ResNet(dropout=0.4).to(device)
    x = model(x)
    print(x.size())
