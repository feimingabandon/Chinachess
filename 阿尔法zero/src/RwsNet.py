import os

import torch
from torch import nn
import torch.nn.functional as F

'''
        在使用中国象棋形成的数据集中，我们使用7通道对象棋中的每一个类型棋子进行分类、原则上是可以分为 14 类
    但为了减少计算量，就将其分为 7 类，使用1和-1作为颜色标志，其次我们需要将对方上一步从哪个位置下到哪个位置也使用
    一个二维数组进行表示，即一个宽9长10的二维数组，原点设置为1，终点设置为-1，其余设置为0、再者，需要将当前是哪一方
    下也使用一个二维数组表示，红下则使用一个全为1的二维数组，黑则是全为-1的。
        至此、我们就有了一个通道数为 9 (7+1+1),长:10，宽:9的张量作为残差神经网络的输入、此外、7 类中每一类使用
    一个通道表示，即如果是马类，则将一个局面类的所有马的位置使用-1/1表示,-1表示黑马，1表示红马，其余位置则使用0表示,
    参考 ChessData 类的示例
    planes：通道数
    stride：卷积核大小
    down_sample：如果需要进行跨层连接则此变量值会上一个卷积层加归一层
'''


# 定义残差块
class BasicBlock(nn.Module):
    def __init__(self, conv, bn, planes, down_sample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv
        self.bn1 = bn
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  # 卷积核步长stride
        self.bn2 = nn.BatchNorm2d(planes)
        self.down_sample = down_sample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.down_sample is not None:
            identity = self.down_sample(x)
        out = out + identity

        return self.relu(out)


# 搭建残差神经网络 输入：N, 9, 10, 9 --> N, C, H, W
class ZouNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 网络第一层卷积等操作，为残差块的输入做准备
        self.conv1 = nn.Conv2d(in_channels=9, out_channels=256, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(256)  # 归一化
        self.relu = nn.ReLU()
        # planes：通道数，搭建残差块,这里只搭建了8个残差块，每个layer方法里面有两个
        self.layer1 = self._make_layer(planes=256)
        self.layer2 = self._make_layer(planes=256)
        self.layer3 = self._make_layer(planes=256)
        self.layer4 = self._make_layer(planes=256)
        # 搭建策略网络输出块
        self.policy_conv = nn.Conv2d(in_channels=256, out_channels=16, kernel_size=(1, 1), stride=(1, 1))
        self.policy_bn = nn.BatchNorm2d(16)
        self.policy_relu = nn.ReLU()
        self.policy_linear = nn.Linear(16 * 9 * 10, 2086)
        # 价值头
        self.value_conv = nn.Conv2d(in_channels=256, out_channels=8, kernel_size=(1, 1), stride=(1, 1))
        self.value_bn = nn.BatchNorm2d(8)
        self.value_relu1 = nn.ReLU()
        self.value_linear1 = nn.Linear(8 * 9 * 10, 256)
        self.value_relu2 = nn.ReLU()
        self.value_linear2 = nn.Linear(256, 1)

        # 定义前向传播
    def forward(self, x):
        # 网络第一层卷积等操作，为残差块的输入做准备
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # 残差块
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # 策略块
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = self.policy_relu(policy)
        # 将张量展平为一个二维向量，-1是让函数自动计算维度的长宽
        policy = torch.reshape(policy, [-1, 16 * 10 * 9])
        # 将张 量转换为一维向量且输出值2086是所有棋子运动的概率
        policy = self.policy_linear(policy)
        # 将2086的输出值转换为log概率值，如果直接输出概率，因为概率是浮点数，而电脑对浮点数的精度会出现问题，后期可以使用exp函数去计算概率值
        policy = F.log_softmax(policy, dim=1)

        # 价值块
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = self.value_relu1(value)
        value = torch.reshape(value, [-1, 8 * 10 * 9])
        value = self.value_linear1(value)
        value = self.value_relu2(value)
        value = self.value_linear2(value)
        value = torch.tanh(value)

        return policy, value

    def _make_layer(self, planes):
        down_sample = nn.Sequential(
            nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes)
        )

        layers = [BasicBlock(
                             nn.Conv2d(in_channels=planes, out_channels=planes, kernel_size=3, stride=1, padding=1, bias=False),
                             nn.BatchNorm2d(planes), planes, down_sample=down_sample),
                  BasicBlock(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False),
                             nn.BatchNorm2d(planes), planes)]  # 存储残差块
        return nn.Sequential(*layers)


# 使用神经网络进行预测
def zou():
    net = ZouNet()
    # 加载模型，先判断是否已经存在模型，如果存在就加载已经存在的模型
    if os.path.isfile("model/zou.pth"):
        # 存在则加载
        net.load_state_dict(torch.load("model/zou.pth"))
    # 判断是否能使用GPU进行训练
    if torch.cuda.is_available():
        net = net.cuda()
    return net


def use_zou(net, stacked_tensor):
    # 判断是否能使用GPU进行加速
    if torch.cuda.is_available():
        stacked_tensor = stacked_tensor.cuda()
    policy, value = net(stacked_tensor)
    return policy, value

# if __name__ == '__main__':
#     net = ZouNet().to('cuda')
#     test_data = torch.ones([1, 9, 10, 9]).to('cuda')
#     print(test_data)
#     x_act, x_val = net(test_data)
#     print(x_act)  # 8, 2086
#     print(x_val)  # 8, 1
#     from train import *
