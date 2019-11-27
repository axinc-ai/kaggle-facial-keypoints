import torch
import torch.nn as nn


# For block
KERNEL_SIZE = 3
STRIDE = 2
PADDING = 1

# For first conv
# if you want to treat RGB, change this value to 3 (didn't test however)
IMG_CHANNEL = 1
FIRST_OUTPUT_CHANNELS = 32
FIRST_KERNEL_SIZE = 3
FIRST_STRIDE = 1

# Global
OUTPUT_CHANNELS = 30  # 30 feature points


class _ResNetBlock(nn.Module):
    def __init__(
            self, in_channels, nb_filters, stride, donwsample=None, dropout=0.0
    ):
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
        self.first_conv = nn.Conv2d(
            in_channels=IMG_CHANNEL,
            out_channels=FIRST_OUTPUT_CHANNELS,
            kernel_size=FIRST_KERNEL_SIZE,
            stride=FIRST_STRIDE,
            padding=PADDING
        )
        self.block_1 = self._make_block(
            FIRST_OUTPUT_CHANNELS,
            FIRST_OUTPUT_CHANNELS * 4,
            STRIDE,
            dropout=dropout
        )
        self.block_2 = self._make_block(
            FIRST_OUTPUT_CHANNELS * 4,
            FIRST_OUTPUT_CHANNELS * 8,
            STRIDE,
            dropout=dropout
        )
        self.block_3 = self._make_block(
            FIRST_OUTPUT_CHANNELS * 8,
            FIRST_OUTPUT_CHANNELS * 16,
            STRIDE,
            dropout=dropout)
        self.block_4 = self._make_block(
            FIRST_OUTPUT_CHANNELS * 16,
            FIRST_OUTPUT_CHANNELS * 16,
            STRIDE,
            dropout=dropout
        )
        self.block_5 = self._make_block(
            FIRST_OUTPUT_CHANNELS * 16,
            FIRST_OUTPUT_CHANNELS * 16,
            STRIDE,
            dropout=dropout
        )

        self.avgpool = nn.AvgPool2d(kernel_size=(3, 3))
        self.last_lin = nn.Sequential(
            nn.Linear(FIRST_OUTPUT_CHANNELS * 16, OUTPUT_CHANNELS * 16),
            nn.Linear(OUTPUT_CHANNELS * 16, OUTPUT_CHANNELS)
        )

    def _make_block(self, in_channels, out_channels, stride, dropout=0.0):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            # TODO do we really need to put batch norm 2d here ?
            downsample = nn.Sequential(
                nn.BatchNorm2d(num_features=in_channels),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride),
            )
        return _ResNetBlock(
            in_channels,
            out_channels,
            stride,
            donwsample=downsample,
            dropout=dropout
        )

    def forward(self, x, verbose=False):
        x = self.first_conv(x)
        if verbose:
            print(x.size())

        x = self.block_1(x)
        if verbose:
            print(x.size())

        x = self.block_2(x)
        if verbose:
            print(x.size())

        x = self.block_3(x)
        if verbose:
            print(x.size())

        x = self.block_4(x)
        if verbose:
            print(x.size())

        x = self.block_5(x)
        if verbose:
            print(x.size())

        x = self.avgpool(x)
        if verbose:
            print("avg_pool")
            print(x.size())

        # reshape (Batch, Channels, 1, 1) to (Batch, Channels)
        x = self.last_lin(x.reshape(x.size(0), -1))
        if verbose:
            print(x.size())
        return x


# for debug purpose
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    x = torch.rand(32, 1, 96, 96).to(device)
    # print(x.size())
    model = ResNet(dropout=0.4).to(device)
    x = model(x, verbose=True)
    # print(x.size())
