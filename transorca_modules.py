import numpy as np

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

# can be set to lower values to decrease memory usage
# at least 4000 * 50 recommended for performance
Blocksize = 4000 * 200


class PTNet(nn.Module):
    def __init__(self, num_1d=None):
        """
        Orca 1Mb model. The trained model weighted can be
        loaded into Encoder and Decoder_1m modules.

        Parameters
        ----------
        num_1d : int or None, optional
            The number of 1D targets used for the auxiliary
            task of predicting ChIP-seq profiles.
        """
        super(PTNet, self).__init__()

        self.lconv1 = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
        )
        self.lconv1.requires_grad_(False)

        self.conv1 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 64, kernel_size=9, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )
        self.conv1.requires_grad_(False)

        self.lconv2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(64, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
        )
        self.lconv2.requires_grad_(False)

        self.conv2 = nn.Sequential(
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
            nn.Conv1d(96, 96, kernel_size=9, padding=4),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )
        self.conv2.requires_grad_(False)

        self.lconv3 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(96, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )
        self.lconv3.requires_grad_(False)

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.conv3.requires_grad_(False)

        self.lconv4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )
        self.lconv4.requires_grad_(False)

        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.conv4.requires_grad_(False)

        self.lconv5 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )
        self.lconv5.requires_grad_(False)

        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.conv5.requires_grad_(False)

        self.lconv6 = nn.Sequential(
            nn.MaxPool1d(kernel_size=5, stride=5),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )
        self.lconv6.requires_grad_(False)

        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.conv6.requires_grad_(False)

        self.lconv7 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
        )
        self.lconv7.requires_grad_(False)

        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=9, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.conv7.requires_grad_(False)

        self.lconvtwos_ut = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Dropout(p=0.1),
                    nn.Conv2d(128, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                ),
            ]
        )

        self.convtwos_ut = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=2, dilation=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=4, dilation=4),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=8, dilation=8),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=16, dilation=16),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=32, dilation=32),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=64, dilation=64),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                ),
            ]
        )
        self.final = nn.Sequential(
            nn.Conv2d(64, 5, kernel_size=(1, 1), padding=0),
            nn.BatchNorm2d(5),
            nn.ReLU(inplace=True),
            nn.Conv2d(5, 1, kernel_size=(1, 1), padding=0),
        )
        if num_1d is not None:
            self.final_1d = nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=1, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, num_1d, kernel_size=1, padding=0),
                nn.Sigmoid(),
            )
        self.num_1d = num_1d

    def forward(self, x):
        """Forward propagation of a batch."""

        def run0(x, dummy):
            lout1 = self.lconv1(x)
            out1 = self.conv1(lout1)
            lout2 = self.lconv2(out1 + lout1)
            out2 = self.conv2(lout2)
            lout3 = self.lconv3(out2 + lout2)
            out3 = self.conv3(lout3)
            lout4 = self.lconv4(out3 + lout3)
            out4 = self.conv4(lout4)
            lout5 = self.lconv5(out4 + lout4)
            out5 = self.conv5(lout5)
            lout6 = self.lconv6(out5 + lout5)
            out6 = self.conv6(lout6)
            lout7 = self.lconv7(out6 + lout6)
            out7 = self.conv7(lout7)
            mat = out7[:, :, :, None] + out7[:, :, None, :]
            cur = mat
            if self.num_1d:
                output1d = self.final_1d(out7)
                return cur, output1d
            else:
                return cur

        dummy = torch.Tensor(1)
        dummy.requires_grad = True
        if self.num_1d:
            cur, output1d = checkpoint(run0, x, dummy)
        else:
            cur = checkpoint(run0, x, dummy)

        def run1(cur):
            first = True
            for lm, m in zip(self.lconvtwos_ut[:7], self.convtwos_ut[:7]):
                if first:
                    cur = lm(cur)

                    first = False
                else:
                    cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run2(cur):
            for lm, m in zip(self.lconvtwos_ut[7:13], self.convtwos_ut[7:13]):
                cur = lm(cur) + cur
                cur = m(cur) + cur
            return cur

        def run3(cur):
            for lm, m in zip(self.lconvtwos_ut[13:], self.convtwos_ut[13:]):
                cur = lm(cur) + cur
                cur = m(cur) + cur

            cur = self.final(cur)
            cur = 0.5 * cur + 0.5 * cur.transpose(2, 3)
            return cur

        cur = checkpoint(run1, cur)
        cur = checkpoint(run2, cur)
        cur = checkpoint(run3, cur)

        if self.num_1d:
            return cur, output1d
        else:
            return cur

