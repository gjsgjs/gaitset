import torch
import torch.nn as nn
import copy
from .basic_blocks import BasicConv2d, HorizontalPoolingPyramid, SeparateFCs, PackSequenceWrapper, SetBlockWrapper


class SetNet(nn.Module):
    def __init__(self, hidden_dim):
        super(SetNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_frame = None

        in_c = [1, 32, 64, 128]
        self.set_block1 = nn.Sequential(BasicConv2d(in_c[0], in_c[1], 5, 1, 2),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[1], in_c[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block2 = nn.Sequential(BasicConv2d(in_c[1], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[2], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block3 = nn.Sequential(BasicConv2d(in_c[2], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[3], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True))
        # MGP这一部分的CNN与池化操作
        self.gl_block2 = copy.deepcopy(self.set_block2)
        self.gl_block3 = copy.deepcopy(self.set_block3)

        self.set_block1 = SetBlockWrapper(self.set_block1)
        self.set_block2 = SetBlockWrapper(self.set_block2)
        self.set_block3 = SetBlockWrapper(self.set_block3)

        self.set_pooling = PackSequenceWrapper(torch.max)

        self.bin_num = [1, 2, 4, 8, 16]  # HPM的5个scale
        self.HPP = HorizontalPoolingPyramid(bin_num=self.bin_num)

        self.Head = SeparateFCs(parts_num=sum(self.bin_num) * 2, in_channels=128, out_channels=hidden_dim)
        self.init_model()

    def init_model(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

    def forward(self, silho, batch_frame=None):

        # import pdb
        # pdb.set_trace()

        x = silho.unsqueeze(1)
        del silho

        outs = self.set_block1(x)
        gl = self.set_pooling(outs, batch_frame, options={"dim": 2})[0]
        gl = self.gl_block2(gl)

        outs = self.set_block2(outs)
        gl = gl + self.set_pooling(outs, batch_frame, options={"dim": 2})[0]
        gl = self.gl_block3(gl)

        outs = self.set_block3(outs)
        outs = self.set_pooling(outs, batch_frame, options={"dim": 2})[0]
        gl = gl + outs

        # Horizontal Pooling Matching, HPM
        feature1 = self.HPP(outs)  # [n, c, p]
        feature2 = self.HPP(gl)  # [n, c, p]
        feature = torch.cat([feature1, feature2], -1)  # [n, c, p]
        embs = self.Head(feature)
        return embs, None
