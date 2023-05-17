import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from ChessData import *
from RwsNet import *


# 训练模型
'''
    1.传入数据集、将数据集转换为张量
        数据集应该是包含了一局棋的所有局面、使用一个二维数组存储一个局面，有多少个局面就有多少个二维数组，
        如 初始局面 与 红炮移动（(1，2)移动到(4,2)，即当头炮）后的局面为：
        state_list_init = [
                  [['红车', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '红炮', '一一', '一一', '一一', '一一', '一一', '红炮', '一一'],
                   ['红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
                   ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑马', '黑车']],
                  [['红车', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '一一', '一一', '一一', '红炮', '一一', '一一', '红炮', '一一'],
                   ['红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
                   ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑马', '黑车']]
                   ]
       我们再定义一个二维数组(state_list_init2)，里面每一个一维数组与上面的数组中的每一个局面一一对应，一维数组里面存放两个值:
       1、对方最近一步移动的位置(如果没有，就用0000代替)
       2、当前是哪个下(1表示红，-1表示黑)
       我们需要对数据进行处理，即通过一个局面去得到对方上一步的落子，是由哪一方下以及棋子 此三者的通道数
       我们再定义一个标准位(flag)来表示谁赢下了比赛(1表示红赢，-1表示黑赢)
       我们再加一个蒙特卡洛树搜索的结果，这样用来训练策略网络，是一个二维数组，里面每一个值都是当前局面下所有落子概率的一个一维数组(长度为2086)

# '''
# state_list_init = [
#     [['红车', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
#      ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
#      ['一一', '红炮', '一一', '一一', '一一', '一一', '一一', '红炮', '一一'],
#      ['红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵'],
#      ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
#      ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
#      ['黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
#      ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '一一'],
#      ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
#      ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑马', '黑车']],
#     [['红车', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
#      ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
#      ['一一', '一一', '一一', '红炮', '一一', '一一', '一一', '红炮', '一一'],
#      ['红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵'],
#      ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
#      ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
#      ['黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
#      ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '一一'],
#      ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
#      ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑马', '黑车']]
# ]
# a = []
# for i in range(2086):
#     if i == 300:
#         a.append(1.0)
#         continue
#     a.append(0.0)
# # 存储对手上一步棋，当前是谁下，当前局面下由蒙特卡洛得到的全部落子概率、以及落子方最终胜利情况
# state_list_init2 = [['0000', '1', a, 1.0], ['2124', '-1', a, -1.0]]


# 将得到的数据转换为张量，值是单精度浮点数,会返回一个数据集和这个数据集长度
def get_date(state_list_init, state_list_init2):

    # 定义一个数组存储由蒙特卡洛搜索获得的下一步(策略网络训练用)，存放单个字符串
    tree_list = []
    # 定义一个数组存储每个局面所对应的胜负(价值网络训练用)，存放单个字符串
    win_fail_list = []
    # 创建一个存储张量集，tree_list，win_fail_list的变量
    dataset = torch.zeros(len(state_list_init), 9, 10, 9)
    # 创建一个索引
    index = 0
    # 遍历局面，将局面处理为9X10X9的通道
    for list_data, state_list in zip(state_list_init, state_list_init2):
        # 获取7通道的棋子的二维数组形式
        tensor = convert_chess_tensor(list_data)
        # 添加对手上一步棋的位置
        if state_list[0] == '0000':
            # 表示没有上一步，则全部设置为0
            tensor[7] = torch.zeros((10, 9))
        else:
            # 表示上一步有落子，则取出，并进行赋值
            np1 = torch.zeros((10, 9))
            str1 = state_list[0]
            np1[int(str1[0])][int(str1[1])] = -1  # 起始
            np1[int(str1[2])][int(str1[3])] = 1  # 结束
            tensor[7] = np1
        # 添加当前是谁下
        if state_list[1] == '1':
            tensor[8] = torch.ones((10, 9))
        else:
            tensor[8] = -1 * torch.ones((10, 9))
        # 将张量添加张量集
        dataset[index] = tensor

        # 存储由蒙特卡洛树得到下此局中此局面的所有落子概率
        tree_list.append(state_list[2])
        # 存储下此局面者最终是否为胜利者的值
        win_fail_list.append(int(state_list[3]))

        print(tensor)

    if len(dataset) < 128:
        batch_size = len(dataset)
    else:
        batch_size = 128
    '''
        这段代码是一个函数，其作用是将数据集、决策树数据和胜负结果数据组合成一个PyTorch的DataLoader对象，以便于模型训练。下面是各个参数的解释：

        dataset：传入的数据集，可以是一个列表、数组等等。
        tree_list：传入的决策树数据，这里假设它是一个列表。
        win_fail_list：传入的胜负结果数据，这里假设它也是一个列表。
        batch_size：批量大小，即每次读入多少个样本进行训练。
        drop_last：如果数据集大小不能被批量大小整除，是否舍弃最后一批数据。
        shuffle：是否对数据进行随机洗牌。
        在函数中，TensorDataset是一个PyTorch提供的类，用于将多个tensor组合成一个数据集，这里将dataset、tree_list、win_fail_list组合成一个数据集。
        最终返回的是一个DataLoader对象，可以方便地对数据进行迭代访问，也就是我们常说的"批量"训练。
    '''
    # 将此张量列表转换为张量集,并返回
    return DataLoader(TensorDataset(dataset, torch.tensor(tree_list, dtype=torch.float32), torch.tensor(win_fail_list, dtype=torch.float32)), batch_size=batch_size, drop_last=True, shuffle=False)


# 将一个局面转换为7个二维数组形式
def convert_chess_tensor(data):
    # 存储7个二维数组，剩下两个后面会添加
    tensor = torch.zeros((9, 10, 9))
    # 先将局面由普通的二维数组转换为numpy类型的二维数组
    data = np.array(data)
    # 将局面转换为张量，有7通道，一个通道占一种棋子
    # 通过值获取二维数组中值的下标(返回元组，若没有返回：[])
    tensor[0] = str_onehot(np.argwhere(data == '黑车'), np.argwhere(data == '红车'))
    tensor[1] = str_onehot(np.argwhere(data == '黑马'), np.argwhere(data == '红马'))
    tensor[2] = str_onehot(np.argwhere(data == '黑炮'), np.argwhere(data == '红炮'))
    tensor[3] = str_onehot(np.argwhere(data == '黑相'), np.argwhere(data == '红相'))
    tensor[4] = str_onehot(np.argwhere(data == '黑士'), np.argwhere(data == '红士'))
    tensor[5] = str_onehot(np.argwhere(data == '黑兵'), np.argwhere(data == '红兵'))
    tensor[6] = str_onehot(np.argwhere(data == '黑帅'), np.argwhere(data == '红帅'))

    return tensor


# 2.训练价值策略网络
def train_zou_net(state_list_init, state_list_init2):
    # 初始化模型结构
    zou = ZouNet()
    # 加载模型，先判断是否已经存在模型，如果存在就加载已经存在的模型
    if os.path.isfile("model/zou.pth"):
        # 存在则加载
        zou.load_state_dict(torch.load("model/zou.pth"))
    else:
        print("不存在模型文件，使用初始化模型...")
    # 判断是否能使用GPU进行训练
    if torch.cuda.is_available():
        zou = zou.cuda()
    # 优化器
    learning_rate = 0.02  # 梯度下降的步长
    optimizer = torch.optim.SGD(zou.parameters(), lr=learning_rate)
    # 设置训练网络的参数
    # 记录训练的次数
    total_train_step = 0
    # 训练的轮数
    epoch = 1
    # 添加 tensorboard ,即训练结果图像化
    writer = SummaryWriter("../logs_train")
    # 获取数据集
    train_dataLoader = get_date(state_list_init, state_list_init2)
    for i in range(epoch):
        print("----------第 {} 轮训练开始------------".format(i))
        # 训练步骤开始
        zou.train()  # 固定写法，用于将模型 zou 切换到训练模式，会对特定网络或参数有加速效果
        for data in train_dataLoader:
            stacked_tensor, tree, win_fail = data
            win_fail = torch.unsqueeze(win_fail, dim=1)
            print("数据集大小：")
            print(stacked_tensor.size())
            print("tree大小：")
            print(tree.size())
            print("win_fail：")
            print(win_fail.size())
            # 判断是否能使用GPU进行加速
            if torch.cuda.is_available():
                stacked_tensor = stacked_tensor.cuda()
                tree = tree.cuda()
                win_fail = win_fail.cuda()
            outputs = zou(stacked_tensor)
            policy, value = outputs
            value_loss = F.mse_loss(input=value, target=win_fail)
            policy_loss = -torch.mean(torch.sum(tree * policy, dim=1))
            # 优化器优化模型
            optimizer.zero_grad()  # 梯度清零、在PyTorch中，每一次计算梯度都会累加到之前的梯度上，因此在每次反向传播前，需要手动清除上一次计算的梯度，避免对本次梯度计算的影响。
            loss = value_loss + policy_loss
            loss.backward()  # 是 PyTorch 中计算张量梯度的函数、根据链式法则自动计算梯度，即将当前的梯度值传递给前面的层，通过链式法则计算出每一层的梯度。
            optimizer.step()  # 使用优化器的 step() 函数根据梯度来更新参数，从而实现梯度下降优化。

            total_train_step += 1  # 训练次数加1

            print("训练次数：{}，value_loss：{}，policy_loss：{}".format(total_train_step, value_loss.item(),policy_loss.item()))
            writer.add_scalar("value_loss",  value_loss.item(), total_train_step)
            writer.add_scalar("policy_loss",  policy_loss.item(), total_train_step)
            # 保存模型(每一轮保存一次)(只保存模型参数，使用字典模式)
            torch.save(zou.state_dict(), "model/zou.pth")
            print("模型已保存")





# train_zou_net()