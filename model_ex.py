# ResNet and Inception named Reception
import torch
from torch import nn as nn
import torch.nn.functional as F
#from torchsummary import summary


class Reception(torch.nn.Module):
    def __init__(self):
        super(Reception, self).__init__()
        self.conv0_1 = nn.Sequential(
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=1, padding=0),
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU()
        )
        self.conv1_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=8, kernel_size=1, padding=0),
            nn.Dropout(0.3),
            nn.BatchNorm1d(8),
            nn.LeakyReLU()
        )
        self.conv2_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=28, kernel_size=5, padding=2),
            nn.Dropout(0.5),
            nn.BatchNorm1d(28),
            nn.LeakyReLU()
        )
        self.conv3_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.conv4_1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=2, kernel_size=1, padding=0),
            nn.BatchNorm1d(2),
            nn.ReLU()
        )
        # self.max_pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.max_pool = nn.AvgPool1d(kernel_size=2,stride=2)

    def forward(self, x):
        #print('reception_in', x.shape)
        x = self.conv0_1(x)
        branch1_1 = self.conv1_1(x)
        branch2_1 = self.conv2_1(x)
        branch3_1 = self.conv3_1(x)
        branch3_1 = self.conv3_1(branch3_1)
        branch3_2 = F.relu(branch3_1)
        branch3_3 = self.conv2_1(branch3_2)
        out = torch.cat([branch1_1, branch2_1, branch3_3], dim=1)
        # print('out', out.shape)
        # print('x', x.shape)
        x = out + x
        out = F.relu(x)
        branch1_1 = self.conv1_1(out)
        branch2_1 = self.conv2_1(out)
        branch3_1 = self.conv3_1(out)
        branch3_1 = self.conv3_1(branch3_1)
        branch3_2 = F.relu(branch3_1)
        branch3_3 = self.conv2_1(branch3_2)
        out = torch.cat([branch1_1, branch2_1, branch3_3], dim=1)  # 32,1,1024
        x = out + x
        out = self.max_pool(x)  # 32,1,512
        out = self.conv4_1(out)  # 2,1,512
        return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        class_num = 3  # 类别数量
        self.re_ception = Reception()
        self.full_connection_1 = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            # nn.Linear(64*6, 64),
            # nn.Dropout(p=0.5),
            # nn.ReLU(),
            nn.Linear(64,class_num)
        )
        self.softmax = nn.Softmax()

    def forward(self, x):
        # print(x.shape)
        out = self.re_ception(x)
        out = self.re_ception(out)  # 2,512
        out = self.re_ception(out)  # 2,256
        out = self.re_ception(out)  # 2,128
        #out = self.re_ception(out)  # 2,64
        # print(f"reception_out{out.shape}")
        linear_out = nn.Flatten()(out)
        out = linear_out
        # out = torch.dropout(linear_out, p=0.3, train=self.training)
        #print('out', out.shape)
        out = self.full_connection_1(out)
        # forecast_ans = self.softmax(out)
        return out
##############################################
class VGG16(nn.Module):

    def __init__(self):
        super(VGG16, self).__init__()

        # 3 * 224 * 224
        self.conv1_1 = nn.Conv1d(2, 64, 3)  # 64 * 222 * 222
        self.conv1_2 = nn.Conv1d(64, 64, 3, padding=1)  # 64 * 222* 222
        self.maxpool1 = nn.MaxPool1d(2, padding=1)  # pooling 64 * 112 * 112

        self.conv2_1 = nn.Conv1d(64, 128, 3)  # 128 * 110 * 110
        self.conv2_2 = nn.Conv1d(128, 128, 3, padding=1)  # 128 * 110 * 110
        self.maxpool2 = nn.MaxPool1d(2, padding=1)  # pooling 128 * 56 * 56

        self.conv3_1 = nn.Conv1d(128, 256, 3)  # 256 * 54 * 54
        self.conv3_2 = nn.Conv1d(256, 256, 3, padding=1)  # 256 * 54 * 54
        self.conv3_3 = nn.Conv1d(256, 256, 3, padding=1)  # 256 * 54 * 54
        self.maxpool3 = nn.MaxPool1d(2, padding=1)  # pooling 256 * 28 * 28

        self.conv4_1 = nn.Conv1d(256, 512, 3)  # 512 * 26 * 26
        self.conv4_2 = nn.Conv1d(512, 512, 3, padding=1)  # 512 * 26 * 26
        self.conv4_3 = nn.Conv1d(512, 512, 3, padding=1)  # 512 * 26 * 26
        self.maxpool4 = nn.MaxPool1d(2, padding=1)  # pooling 512 * 14 * 14

        self.conv5_1 = nn.Conv1d(512, 512, 3)  # 512 * 12 * 12
        self.conv5_2 = nn.Conv1d(512, 512, 3, padding=1)  # 512 * 12 * 12
        self.conv5_3 = nn.Conv1d(512, 512, 3, padding=1)  # 512 * 12 * 12
        self.maxpool5 = nn.MaxPool1d(2, padding=1)  # pooling 512 * 7 * 7

        # view

        self.fc1 = nn.Linear(16384, 4096)
        self.fc2 = nn.Linear(4096, 100)
        self.fc3 = nn.Linear(100, 3)
        # softmax 1 * 1 * 1000

    def forward(self, x):
        # x.size(0)即为batch_size
        in_size = x.size(0)

        out = self.conv1_1(x)  # 222
        out = F.relu(out)
        out = self.conv1_2(out)  # 222
        out = F.relu(out)
        out = self.maxpool1(out)  # 112

        out = self.conv2_1(out)  # 110
        out = F.relu(out)
        out = self.conv2_2(out)  # 110
        out = F.relu(out)
        out = self.maxpool2(out)  # 56

        out = self.conv3_1(out)  # 54
        out = F.relu(out)
        out = self.conv3_2(out)  # 54
        out = F.relu(out)
        out = self.conv3_3(out)  # 54
        out = F.relu(out)
        out = self.maxpool3(out)  # 28

        out = self.conv4_1(out)  # 26
        out = F.relu(out)
        out = self.conv4_2(out)  # 26
        out = F.relu(out)
        out = self.conv4_3(out)  # 26
        out = F.relu(out)
        out = self.maxpool4(out)  # 14

        out = self.conv5_1(out)  # 12
        out = F.relu(out)
        out = self.conv5_2(out)  # 12
        out = F.relu(out)
        out = self.conv5_3(out)  # 12
        out = F.relu(out)
        out = self.maxpool5(out)  # 7
        # print(f"reception_out{out.shape}")
        # 展平
        out = out.view(in_size, -1)

        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)

        out = F.log_softmax(out, dim=1)

        return out
#####################################################
class RestNetBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        output = self.conv1(x)
        output = F.relu(self.bn1(output))
        output = self.conv2(output)
        output = self.bn2(output)
        return F.relu(x + output)


class RestNetDownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(RestNetDownBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride[0], padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride[1], padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.extra = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride[0], padding=0),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x):
        extra_x = self.extra(x)
        output = self.conv1(x)
        out = F.relu(self.bn1(output))
        out = self.conv2(out)
        out = self.bn2(out)
        return F.relu(extra_x + out)


class RestNet18(nn.Module):
    def __init__(self):
        super(RestNet18, self).__init__()
        self.conv1 = nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(RestNetBasicBlock(64, 64, 1),
                                    RestNetBasicBlock(64, 64, 1))
        self.layer2 = nn.Sequential(RestNetDownBlock(64, 128, [2, 1]),
                                    RestNetBasicBlock(128, 128, 1))
        self.layer3 = nn.Sequential(RestNetDownBlock(128, 256, [2, 1]),
                                    RestNetBasicBlock(256, 256, 1))
        self.layer4 = nn.Sequential(RestNetDownBlock(256, 512, [2, 1]),
                                    RestNetBasicBlock(512, 512, 1))
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(512, 3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out

############################################################
class Bottleneck(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1,1],padding=[0,1,0],first=False) -> None:
        super(Bottleneck,self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels,out_channels,kernel_size=1,stride=stride[0],padding=padding[0],bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv1d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding[1],bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv1d(out_channels,out_channels*4,kernel_size=1,stride=stride[2],padding=padding[2],bias=False),
            nn.BatchNorm1d(out_channels*4),
            nn.Dropout(0.5)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if first:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                # 注意跳变时 都是stride==2的时候 也就是每次输出信道升维的时候
                nn.Conv1d(in_channels, out_channels*4, kernel_size=1, stride=stride[1], bias=False),
                nn.BatchNorm1d(out_channels*4)
            )
    def forward(self, x):
        out = self.bottleneck(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet50(nn.Module):
    def __init__(self, Bottleneck, num_classes=4) -> None:
        super(ResNet50, self).__init__()
        self.in_channels = 64
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        # conv2
        self.conv2 = self._make_layer(Bottleneck, 64, [[1, 1, 1]] * 3, [[0, 1, 0]] * 3)

        # conv3
        self.conv3 = self._make_layer(Bottleneck, 128, [[1, 2, 1]] + [[1, 1, 1]] * 3, [[0, 1, 0]] * 4)

        # conv4
        self.conv4 = self._make_layer(Bottleneck, 256, [[1, 2, 1]] + [[1, 1, 1]] * 5, [[0, 1, 0]] * 6)

        # conv5
        self.conv5 = self._make_layer(Bottleneck, 512, [[1, 2, 1]] + [[1, 1, 1]] * 2, [[0, 1, 0]] * 3)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(2048,200),
            nn.Dropout(0.5),
            nn.Linear(200, num_classes)
        )

    def _make_layer(self, block, out_channels, strides, paddings):
        layers = []
        # 用来判断是否为每个block层的第一层
        flag = True
        for i in range(0, len(strides)):
            layers.append(block(self.in_channels, out_channels, strides[i], paddings[i], first=flag))
            flag = False
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)

        out = self.avgpool(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        return out
#########################################################
class CommonBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(CommonBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        identity = x

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)

class SpecialBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(SpecialBlock, self).__init__()
        self.change_channel = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride[0], padding=0, bias=False),
            nn.BatchNorm1d(out_channel)
        )
        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride[0], padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=stride[1], padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)

    def forward(self, x):
        identity = self.change_channel(x)

        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))

        x += identity
        return F.relu(x, inplace=True)


class ResNet34(nn.Module):
    def __init__(self, classes_num=4):
        super(ResNet34, self).__init__()
        self.prepare = nn.Sequential(
            nn.Conv1d(5, 64, 7, 2, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, 2, 1)
        )
        self.layer1 = nn.Sequential(
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1),
            CommonBlock(64, 64, 1)
        )
        self.layer2 = nn.Sequential(
            SpecialBlock(64, 128, [2, 1]),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1),
            CommonBlock(128, 128, 1)
        )
        self.layer3 = nn.Sequential(
            SpecialBlock(128, 256, [2, 1]),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1),
            CommonBlock(256, 256, 1)
        )
        self.layer4 = nn.Sequential(
            SpecialBlock(256, 512, [2, 1]),
            CommonBlock(512, 512, 1),
            CommonBlock(512, 512, 1)
        )
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, classes_num)
        )

    def forward(self, x):
        x = self.prepare(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

if __name__ == '__main__':
    module = ResNet50(Bottleneck)
    print(module)

    data = torch.randn([2,2, 1024])
    print(f"data_shape{data.shape}")
    x = module(data)
    print(x)
    print(f"x_shape{x.shape}")
    # print(data.shape)
    # module(data)

    # summary(module, (2, 1024))
