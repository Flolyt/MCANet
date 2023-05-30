import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from .hardnet_85 import hardnet
from .modules.context_module import *
from .modules.attention_module import *
from .modules.decoder_module import *

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = Conv(inp_dim, int(out_dim / 2), 1, relu=False)
        self.bn2 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv2 = Conv(int(out_dim / 2), int(out_dim / 2), 3, relu=False)
        self.bn3 = nn.BatchNorm2d(int(out_dim / 2))
        self.conv3 = Conv(int(out_dim / 2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)


class Fusion_block(nn.Module):
    def __init__(self, channel, r_2, drop_rate=0.3):
        super(Fusion_block, self).__init__()

        # channel attention for decoder, use SE Block
        self.fc1 = nn.Conv2d(channel, channel // r_2, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channel // r_2, channel, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_g1 = Conv(32, channel, 1, bn=True, relu=False)

        # bi-linear modelling for both
        self.W_g = Conv(channel, channel//3, 1, bn=True, relu=False)
        self.W_x = Conv(channel, channel//3, 1, bn=True, relu=False)
        self.W_pred = Conv(channel, channel//3, 1, bn=True, relu=False)
        self.foregound_conv = Conv(channel, channel // 3, 1, bn=True, relu=False)
        self.W = Conv(channel//3, channel, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(channel + channel + channel, channel)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate
        # self.conv2 = nn.Conv2d(channel, out_channel, kernel_size=kernel_size,
        #                        stride=stride, padding=padding)
        # self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, g, x, pred):
        # bilinear pooling
        # g = self.W_g1(g)
        W_g = self.W_g(g)
        # print(W_g.shape)
        W_x = self.W_x(x)
        # print(W_x.shape)
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        boundary_att = 1 - (dist / 0.5)
        W_pred1 = g * boundary_att
        W_pred1 = self.W_pred(W_pred1)
        # print(W_pred1.shape)
        bp = self.W(W_g * W_x * W_pred1)

        # spatial attention for encoder
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in

        # channel attetion for decoder
        x_in = x
        x = x.mean((2, 3), keepdim=True)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * x_in
        fuse = self.residual(torch.cat([g, x, bp], 1))


        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse



class Fusion_block2(nn.Module):
    def __init__(self, channel, drop_rate=0.3):
        super(Fusion_block2, self).__init__()
        self.sigmoid = nn.Sigmoid()
        # spatial attention for F_l
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, 7, bn=True, relu=False, bias=False)
        self.W_g1 = Conv(channel, channel, 1, bn=True, relu=False)

        # bi-linear modelling for both
        self.W_g = Conv(channel, channel//3, 1, bn=True, relu=False)
        self.W_pred = Conv(channel, channel//3, 1, bn=True, relu=False)
        self.foregound_conv = Conv(channel, channel // 3, 1, bn=True, relu=False)
        self.W = Conv(channel//3, channel, 3, bn=True, relu=True)

        self.relu = nn.ReLU(inplace=True)

        self.residual = Residual(channel + channel , channel)

        self.dropout = nn.Dropout2d(drop_rate)
        self.drop_rate = drop_rate
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, g, pred):
        # bilinear pooling
        W_g = self.W_g(g)
        # print(W_x.shape)
        score = torch.sigmoid(pred)
        dist = torch.abs(score - 0.5)
        boundary_att = 1 - (dist / 0.5)
        W_pred1 = g * boundary_att
        W_pred1 = self.W_pred(W_pred1)
        # print(W_pred1.shape)
        bp = self.W(W_g * W_pred1)

        # spatial attention for encoder
        g_in = g
        g = self.compress(g)
        g = self.spatial(g)
        g = self.sigmoid(g) * g_in
        fuse = self.residual(torch.cat([g, bp], 1))
        if self.drop_rate > 0:
            return self.dropout(fuse)
        else:
            return fuse

class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.key_conv = Conv(in_channels, in_channels // 8, kSize=(1, 1),stride=1,padding=0)
        self.value_conv = Conv(in_channels, in_channels, kSize=(1, 1),stride=1,padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out


class aggregation2(nn.Module):
    def __init__(self, channel):
        super(aggregation2, self).__init__()
        self.x5_x4 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                   nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                   nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                   nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                   nn.ReLU(inplace=True))
        self.x5_x4_x3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                      nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                      nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                      nn.ReLU(inplace=True))
        self.x5_x4_x3_x2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                         nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                         nn.ReLU(inplace=True))
        self.x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                            nn.ReLU(inplace=True))
        self.level = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                   nn.ReLU(inplace=True))
        self.output4 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                     nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                     nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                     nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                     nn.ReLU(inplace=True))
        self.output0 = nn.Sequential(nn.Conv2d(channel, channel//4, kernel_size=3, padding=1), nn.BatchNorm2d(channel//4),
                                     nn.ReLU(inplace=True))
        self.output00 = nn.Sequential(nn.Conv2d(channel // 4, 1, kernel_size=3, padding=1))
        self.output = nn.Sequential(nn.Conv2d(2*channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                     nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(0.5)
        self.Hattn = self_attn(channel, mode='h')
        self.Wattn = self_attn(channel, mode='w')
    def forward(self, x1, x2, x3, x4, x5):
        x5_4 = self.x5_x4(abs(F.upsample(x5, size=x4.size()[2:], mode='bilinear') - x4))
        x4_3 = self.x4_x3(abs(F.upsample(x4, size=x3.size()[2:], mode='bilinear') - x3))
        x3_2 = self.x3_x2(abs(F.upsample(x3, size=x2.size()[2:], mode='bilinear') - x2))
        x2_1 = self.x2_x1(abs(F.upsample(x2, size=x1.size()[2:], mode='bilinear') - x1))
        x5_4_3 = self.x5_x4_x3(abs(F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear') - x4_3))
        x4_3_2 = self.x4_x3_x2(abs(F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear') - x3_2))
        x3_2_1 = self.x3_x2_x1(abs(F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear') - x2_1))
        x5_4_3_2 = self.x5_x4_x3_x2(abs(F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear') - x4_3_2))
        x4_3_2_1 = self.x4_x3_x2_x1(abs(F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear') - x3_2_1))
        x5_4_3_2_1 = self.x5_x4_x3_x2_x1(abs(F.upsample(x5, size=x4_3_2_1.size()[2:], mode='bilinear') - x4_3_2_1))
        level1 = self.level(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)
        level2 = self.level(x3_2 + x4_3_2 + x5_4_3_2)
        level3 = self.level(x4_3 + x5_4_3)
        level4 = x5_4
        output4 = self.output3(F.upsample(x5, size=level3.size()[2:], mode='bilinear') + level4)
        output3 = self.output3(F.upsample(output4, size=level3.size()[2:], mode='bilinear') + level3)
        output2 = self.output2(F.upsample(output3, size=level2.size()[2:], mode='bilinear') + level2)
        output1 = self.output1(F.upsample(output2, size=level1.size()[2:], mode='bilinear') + level1)
        output1 = self.dropout(output1)
        # outputh = self.Hattn(output1)
        # outputw = self.Hattn(output1)
        # output = self.output(torch.cat((outputh, outputw), 1))
        # output = self.output3(output)
        # output = self.output2(output)
        output = self.output0(output1)
        output = self.output00(output)
        return output

class aggregation3(nn.Module):
    def __init__(self, channel):
        super(aggregation3, self).__init__()
        self.x4_x3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                   nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                   nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                   nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                      nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                      nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                         nn.ReLU(inplace=True))
        self.level = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                   nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                     nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                     nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                     nn.ReLU(inplace=True))
        self.output0 = nn.Sequential(nn.Conv2d(channel, channel//4, kernel_size=3, padding=1), nn.BatchNorm2d(channel//4),
                                     nn.ReLU(inplace=True))
        self.output00 = nn.Sequential(nn.Conv2d(channel // 4, 1, kernel_size=3, padding=1))
        self.output = nn.Sequential(nn.Conv2d(2*channel, channel, kernel_size=3, padding=1), nn.BatchNorm2d(channel),
                                     nn.ReLU(inplace=True))
        self.dropout = nn.Dropout2d(0.5)
        self.Hattn = self_attn(channel, mode='h')
        self.Wattn = self_attn(channel, mode='w')
    def forward(self, x1, x2, x3, x4):
        x4_3 = self.x4_x3(abs(F.upsample(x4, size=x3.size()[2:], mode='bilinear') - x3))
        x3_2 = self.x3_x2(abs(F.upsample(x3, size=x2.size()[2:], mode='bilinear') - x2))
        x2_1 = self.x2_x1(abs(F.upsample(x2, size=x1.size()[2:], mode='bilinear') - x1))
        x4_3_2 = self.x4_x3_x2(abs(F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear') - x3_2))
        x3_2_1 = self.x3_x2_x1(abs(F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear') - x2_1))
        x4_3_2_1 = self.x4_x3_x2_x1(abs(F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear') - x3_2_1))
        level1 = self.level(x2_1 + x3_2_1 + x4_3_2_1 )
        level2 = self.level(x3_2 + x4_3_2)
        level3 = x4_3
        output3 = self.output3(F.upsample(x4, size=level3.size()[2:], mode='bilinear') + level3)
        output2 = self.output2(F.upsample(output3, size=level2.size()[2:], mode='bilinear') + level2)
        output1 = self.output1(F.upsample(output2, size=level1.size()[2:], mode='bilinear') + level1)
        output1 = self.dropout(output1)
        # outputh = self.Hattn(output1)
        # outputw = self.Hattn(output1)
        # output = self.output(torch.cat((outputh, outputw), 1))
        # output = self.output3(output)
        # output = self.output2(output)
        output = self.output0(output1)
        output = self.output00(output)
        return output


class ASPP(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ASPP, self).__init__()
        self.relu = nn.ReLU(True)
        self.xx1_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=6, dilation=6),
                                          nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.xx2_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=12, dilation=12),
                                          nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.xx3_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=18, dilation=18),
                                          nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.xx4_conv = nn.Sequential(nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
                                          nn.BatchNorm2d(out_channel), nn.ReLU(inplace=True))
        self.xx5_conv = nn.Conv2d(out_channel * 4 + in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv_res = nn.Conv2d(out_channel, out_channel, kernel_size=1)
        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')
        self.output = nn.Sequential(nn.Conv2d(2 * out_channel, out_channel, kernel_size=3, padding=1), nn.BatchNorm2d(out_channel),
                                    nn.ReLU(inplace=True))
        self.output0 = nn.Sequential(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        # self.dropout = nn.Dropout2d(0.5)
    def forward(self, x):
        x11 = self.xx1_conv(x)
        # print("x11 size:", x11.shape)
        x12 = self.xx2_conv(x)
        # print("x12 size:", x12.shape)
        x13 = self.xx3_conv(x)
        # print("x13 size:", x13.shape)
        x14 = self.xx4_conv(x)
        # print("x14 size:", x14.shape)
        x = torch.cat((x, x11, x12, x13, x14), 1)
        # print("x size:", x.shape)
        x = self.xx5_conv(x)
        outputh = self.Hattn(x)
        outputw = self.Wattn(x)
        output = self.output(torch.cat((outputh, outputw), 1))
        # output = self.Dropout2d(output)
        # output = self.output0(output)
        # print("output size:", output.shape)
        # x = self.xx5_conv(output)
        # x = self.relu(x + self.conv_res(x))
        return output


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1) #卷积核为3 padding为1
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class SideoutBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SideoutBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.dropout = nn.Dropout2d(0.3)

        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class self_attn(nn.Module):
    def __init__(self, in_channels, mode='hw'):
        super(self_attn, self).__init__()

        self.mode = mode

        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1),stride=1,padding=0)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=(1, 1),stride=1,padding=0)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=(1, 1),stride=1,padding=0)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        batch_size, channel, height, width = x.size()

        axis = 1
        if 'h' in self.mode:
            axis *= height
        if 'w' in self.mode:
            axis *= width

        view = (batch_size, -1, axis)

        projected_query = self.query_conv(x).view(*view).permute(0, 2, 1)
        projected_key = self.key_conv(x).view(*view)

        attention_map = torch.bmm(projected_query, projected_key)
        attention = self.sigmoid(attention_map)
        projected_value = self.value_conv(x).view(*view)

        out = torch.bmm(projected_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channel, height, width)

        out = self.gamma * out + x
        return out


class AA_kernel(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(AA_kernel, self).__init__()
        self.conv0 = nn.Conv2d(in_channel, out_channel, kernel_size=1,stride=1,padding=0)
        self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size=(3, 3),stride = 1, padding=1)
        self.Hattn = self_attn(out_channel, mode='h')
        self.Wattn = self_attn(out_channel, mode='w')

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)

        Hx = self.Hattn(x)
        Wx = self.Wattn(Hx)

        return Wx


class h_swish(nn.Module):
    def __init__(self, inplace = True):
        super(h_swish,self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        sigmoid = self.relu(x + 3) / 6
        x = x * sigmoid
        return x



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, padding, stride=1,):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.conv2 =  nn.Conv2d(in_channels // 4, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        self.g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                          kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                               kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, nn.MaxPool2d(kernel_size=(2, 2)))
            self.phi = nn.Sequential(self.phi, nn.MaxPool2d(kernel_size=(2, 2)))

    def forward(self, x):

        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class CCM(nn.Module):
    def __init__(self, in_channels, out_channels,pool_size = [1, 3, 5],in_channel_list=[],
                 out_channel_list = [256, 128/2],cascade=False):
        super(CCM, self).__init__()
        self.cascade=cascade
        self.in_channel_list=in_channel_list
        self.out_channel_list = out_channel_list
        upsampe_scale = [2,4,8,16]
        GClist = []
        GCoutlist = []

        for ps in pool_size:
            GClist.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, out_channels, 1, 1),
                nn.ReLU(inplace=True)))
        GClist.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            NonLocalBlock(out_channels)))
        self.GCmodule = nn.ModuleList(GClist)
        for i in range(len(self.out_channel_list)):
            GCoutlist.append(nn.Sequential(nn.Conv2d(out_channels * 4, self.out_channel_list[i], 3, 1, 1),
                                           nn.BatchNorm2d(self.out_channel_list[i]),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout2d(),
                                           nn.Upsample(scale_factor=upsampe_scale[i], mode='bilinear')))
        self.GCoutmodel = nn.ModuleList(GCoutlist)


    def forward(self, x,y=None):
        xsize = x.size()[2:]
        global_context = []
        for i in range(len(self.GCmodule) - 1):
          global_context.append(F.interpolate(self.GCmodule[i](x), xsize, mode='bilinear', align_corners=True))
        global_context.append(self.GCmodule[-1](x))
        global_context = torch.cat(global_context, dim=1)
        output = []
        for i in range(len(self.GCoutmodel)):
            out=self.GCoutmodel[i](global_context)
            if self.cascade is True and y is not None:
              out=out+y[i]
            output.append(out)
        return output



class mca(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32):
        super(mca, self).__init__()
        # ---- ResNet Backbone ----
        # ---- Receptive Field Block like module ----
        self.se1 = SELayer(192)
        self.se2 = SELayer(320)
        self.se3 = SELayer(480)
        self.se4 = SELayer(1280)
        # self.se5 = SELayer(1280)
        # self.aa_kernel_1 = AA_kernel(320, 320)
        # self.aa_kernel_2 = AA_kernel(640, 640)
        # self.aa_kernel_3 = AA_kernel(1024, 1024)
        self.ccm4 = CCM(640, 32, pool_size=[2, 6, 10], in_channel_list=[128, 64, 64], out_channel_list=[128, 64, 64],
                        cascade=True)
        self.ccm3 = CCM(480, 16, pool_size=[3, 9, 15], in_channel_list=[64, 64], out_channel_list=[64, 64],
                        cascade=True)
        self.ccm2 = CCM(128, 16, pool_size=[4, 12, 20], in_channel_list=[64], out_channel_list=[64], cascade=True)

        # self.decoder5 = DecoderBlock(in_channels=1280, out_channels=720, kernel_size=5, padding=2)
        self.decoder4 = DecoderBlock(in_channels=1280, out_channels=480, kernel_size=5, padding=2)
        self.decoder3 = DecoderBlock(in_channels=480, out_channels=320, kernel_size=3, padding=1)
        self.decoder2 = DecoderBlock(in_channels=320, out_channels=192, kernel_size=1, padding=0)
        self.decoder1 = DecoderBlock(in_channels=192, out_channels=1, kernel_size=1, padding=0)
        # self.decoder2 = DecoderBlock(in_channels=256, out_channels=64)
        # self.decoder1 = DecoderBlock(in_channels=192, out_channels=64)
        self.sideout4 = SideoutBlock(480, 1)
        self.sideout3 = SideoutBlock(320, 1)
        self.sideout2 = SideoutBlock(192, 1)
        self.rfb1_1 = ASPP(in_channel=192, out_channel=channel)
        self.rfb2_1 = ASPP(in_channel=320, out_channel=channel)
        self.rfb3_1 = ASPP(in_channel=480, out_channel=channel)
        self.rfb4_1 = ASPP(in_channel=1280, out_channel=channel)
        # self.rfb5_1 = ASPP(in_channel=1280, out_channel=channel)
        # self.fusion5 = Fusion_block2(1280)
        self.fusion4 = Fusion_block2(1280)
        self.fusion3 = Fusion_block(480,1)
        self.fusion2 = Fusion_block(320,1)
        self.fusion1 = Fusion_block(192,1)

        # ---- Partial Decoder ----
        self.agg = aggregation2(channel)
        self.agg2 = aggregation3(channel)
        self.ra4_conv1 = BasicConv2d(1024, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1, kernel_size=1)
        # ---- reverse attention branch 3 ----
        # 修改第一个channel与rfb3_1一致
        self.ra3_conv1 = BasicConv2d(640, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)
        # ---- reverse attention branch 2 ----
        # 修改第一个channel与rfb4_1一致
        self.ra2_conv1 = BasicConv2d(320, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

        self.hardnet = hardnet(arch=85)


    def forward(self, x):
        # print("x size:", x.shape)
        hardnetout = self.hardnet(x)
        print(hardnetout)
        x1 = hardnetout[0]
        print("x1 size:", x1.shape)
        x1 = self.se1(x1)

        x2 = hardnetout[1]
        print("x2 size:", x2.shape)
        x2 = self.se2(x2)

        x3 = hardnetout[2]
        print("x3 size:", x3.shape)
        x3 = self.se3(x3)

        x4 = hardnetout[3]
        print("x4.size", x4.shape)
        x4 = self.se4(x4)
        # print(hardnetout[4])
        # x5 = hardnetout[4]
        # print("x4.size", x4.shape)
        # x4 = self.se5(x4)

        x1_rfb = self.rfb1_1(x1)
        x2_rfb = self.rfb2_1(x2)  # channel -> 32
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)  # channel -> 32
        # x5_rfb = self.rfb5_1(x5)
        fuse234 = self.agg2(x1_rfb, x2_rfb, x3_rfb, x4_rfb)  # 1 88 88

        # print("fuse234:", fuse234.shape)
        lateral_map_4 = F.interpolate(fuse234, scale_factor=0.125, mode='bilinear')
        x4_fusion = self.fusion4(x4, lateral_map_4)
        d4 = self.decoder4(x4_fusion)
        # print("d4.size", d4.shape)
        lateral_map4 = self.sideout4(d4)

        lateral_map_3 = F.interpolate(fuse234, scale_factor=0.25, mode='bilinear')
        x3_fusion = self.fusion3(x3, d4, lateral_map_3)
        d3 = self.decoder3(x3_fusion)
        lateral_map3 = self.sideout3(d3)
        # print("lateral_map3.size", lateral_map3.shape)
        # print(fuse234.shape)
        # lateral_map_2 = F.interpolate(lateral_map3, scale_factor=2, mode='bilinear')
        lateral_map_2 = F.interpolate(fuse234, scale_factor=0.5, mode='bilinear')
        x2_fusion = self.fusion2(x2, d3, lateral_map_2)
        d2 = self.decoder2(x2_fusion)
        lateral_map2 = self.sideout2(d2)


        lateral_map4 = F.interpolate(lateral_map4, scale_factor=16, mode='bilinear')
        lateral_map3 = F.interpolate(lateral_map3, scale_factor=8, mode='bilinear')
        lateral_map2 = F.interpolate(lateral_map2, scale_factor=4, mode='bilinear')

        return lateral_map4, lateral_map3, lateral_map2



if __name__ == '__main__':
    ras = mca().cuda()
    summary(ras, (3, 352, 352))
    input_tensor = torch.randn(1, 3, 352, 352).cuda()

    out = ras(input_tensor)
