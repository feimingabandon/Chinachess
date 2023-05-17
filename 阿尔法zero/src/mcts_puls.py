
"""
输入一个局面输出一个动作
"""
import copy
import torch
from torch import exp

from train import convert_chess_tensor
from RwsNet import zou, use_zou
from GoOnList import piece_moves_all
from game import legal_all_chess, move_chess, win_fill_draw, out_game
from GoOnList import MyClass

"""
    MCTSTree：表示节点，包含节点需要的结构
"""


class MCTSTree:

    def __init__(self, data=None, parent=0, index=0, play=0):

        # 存储局面
        self.data = data
        # 存储父节点索引
        self.parent = parent
        # 存储子节点集合(字典形式，动作：索引)
        self.child = {}
        # 访问次数
        self.c = 0
        # 存储价值
        self.value = 0
        # 存储所有可能落子
        self.policy = {}
        # 存储当前节点的概率，即策略
        self.policy_my = 0
        # 由原点移动到当前局面的位置信息(如:0010这样)
        self.y_x = '0000'
        # 自身所在的索引
        self.i = index
        # 录入局面的落子方
        self.play = play


"""
实现蒙特卡洛树搜索
"""


class MCTSPlus:

    def __init__(self, situation, play, count_mcts=100, y_x_list=None, z=None):

        # 创建一个变量存储录入的局面
        self.situation = situation
        # 创建一个变量存储蒙特卡洛已经搜索次数
        self.count_mcts_already = 1
        # 存储所有节点
        self.all_node = []
        # 录入局面的落子方
        self.play = play
        # 蒙特卡洛搜索次数
        self.count_mcts = count_mcts
        # 存储最近的四步真实落子
        self.y_x_list = y_x_list
        # 存储神经网络
        self.z = z

    def mcts_train(self):

        """
        1、先通过UCT（置信上限）选择值最大的落子作为子节点
        2、以子节点开始模拟落子，默认次数为20次、模拟时不断利用神经网络的策略与价值选择值最大的进行落子
           直到到达20次或终局，
        3、反向传播，此时子节点被访问次数加1，同时更新其价值、若到达终局，价值则使用胜负即1或-1表示
        4、继续上述选择，重新计算UCT值并重新选择节点并创建与模拟，直至达到设置值：count_mcts
        :return: 返回根节点下的子节点所有动作及其被访问次数（使用字典），再选择访问次数最大的作为真实落子
        """
        situation = copy.deepcopy(self.situation)
        # 先创建根节点
        self.all_node.append(MCTSTree(data=situation, parent=-1, index=0, play=self.play))
        # 将局面转换为神经网络的输入格式
        tensor = get_date_one(situation, self.play, '0000')

        p, v = use_zou(self.z, tensor)
        p = exp(p)
        # 筛选合法落子
        # 将p由张量格式转换为列表，将p与全部合法落子结合形成字典
        policy = dict(zip(piece_moves_all, p.tolist()[0]))
        # 生成当前局面的全部合法落子
        moves_all = legal_all_chess(situation, self.play)
        # 通过以上两者最终确定合法落子与其概率
        for g in moves_all:
            self.all_node[0].policy[g] = policy[g]
        # 将价值传入节点
        self.all_node[0].value = v.item()
        print("蒙特卡洛搜索总次数：{}".format(self.count_mcts))
        print("已经搜索次数：{} -> ".format(self.count_mcts_already), end='')

        for i in range(self.count_mcts):
            #MyClass.qj_count += 1
            # 每次搜索都需要从节点开始
            obj = self.all_node[0]
            # 循环结束标准位（达到50或终局）
            flag = True
            # 寻找未访问的节点，若最终寻找到的节点为终局，则直接进行反向传播
            while flag:
                # 计算置信上限UCT值选择最大的作为动作
                y_x = self.uct_max(copy.deepcopy(obj.policy), copy.deepcopy(obj.child))
                # 判断输出是否为False，是则直接进行反向传播
                if not y_x:
                    print("y_x为False，：循环次数为：{}".format(i), end='')
                    flag = False
                    obj.value += 1
                    obj.c += 1
                    self.back(obj)
                    continue
                # 判断动作是否已经在 child 中，若不在则创建并模拟，若在则继续搜索此动作节点中合法动作的UCT值
                if y_x not in list(obj.child.keys()):
                    # 创建动作对应的节点
                    # 使用深拷贝创建一个新的二维数组的列表
                    new_array = copy.deepcopy(obj.data)
                    move_data = move_chess(new_array, y_x, list(obj.policy.keys()))
                    obj_play = -obj.play
                    child_obj = MCTSTree(data=move_data, parent=obj.i, index=len(self.all_node), play=obj_play)
                    child_obj.c += 1
                    self.all_node.append(child_obj)
                    # 将子节点存储进父节点中
                    obj.child[y_x] = child_obj.i
                    # 将动作放入节点
                    child_obj.y_x = y_x
                    # 判断是否产生胜者
                    win = win_fill_draw(child_obj.data)
                    if win != 0:
                        print("有胜者出现：{} ".format(win), end='')
                        if win == -1:
                            win = 1
                        # 表示有胜者出现
                        child_obj.value = win

                        # 反向传播
                        self.back(child_obj)
                        flag = False
                    else:
                        """
                            未出现胜者则进行扩展,使用for循环一定次数，每次选择UCT最大，记录价值
                            先使用神经网络预测得到价值与策略
                        """
                        # 预测并将值传递到节点
                        p1, v1 = net_result(child_obj.data, child_obj.play, child_obj.y_x, self.z)
                        child_obj.policy = p1
                        child_obj.value = v1.item()
                        # 复制一份局面
                        child_obj_data = copy.deepcopy(child_obj.data)
                        # 存储所有价值
                        value_list = []
                        # 存储play
                        extend_play = child_obj.play
                        # 存储中间值
                        policy, child = p1, {}
                        # 使play和v值反转
                        extend_flag = obj_play
                        ji = 0
                        # 进行扩展
                        for j in range(20):
                            ji += 1
                            # 计算置信上限UCT值选择最大的作为动作
                            y_x = self.uct_max(policy, child)
                            if not y_x:
                                # 直接进行反向传播
                                average = 0
                                for h in value_list:
                                    average += h
                                child_obj.value = (average / len(value_list)) / 2

                                break
                            # 移动局面
                            move_chess(child_obj_data, y_x, list(policy.keys()))
                            # 判断是否为终局
                            win = win_fill_draw(child_obj.data)
                            if win != 0:
                                print("模拟时出现终局：{} ".format(win), end='')
                                child_obj.value = win

                                break
                            # 计算策略与价值
                            extend_flag = -extend_flag
                            policy, v2 = net_result(child_obj_data, extend_flag*extend_play, y_x, self.z)
                            value_list.append(extend_flag*(v2.item()))
                        if ji >= 20:
                            # 说明模拟次数结束也未出现终局，则取价值平均值
                            # 直接进行反向传播
                            average = 0
                            for h in value_list:
                                average += h
                            child_obj.value = average / len(value_list) / 2
                        self.back(child_obj)
                        flag = False

                else:
                    obj = self.all_node[obj.child[y_x]]
            # 将搜索次数加 1
            self.count_mcts_already += 1
            if self.count_mcts_already % 50 == 0:
                print("{} ->".format(self.count_mcts_already), end='')
        s_final, use_count = self.final_max_count(self.all_node[0])
        print('')
        print("可选动作为：{}".format(use_count))
        print("蒙特卡洛树搜索完成、最终选择的动作为：[{}:{}次],移动棋子：{}".format(s_final[0], s_final[1], self.all_node[0].data[int((s_final[0])[0])][int((s_final[0])[1])]))
        return s_final[0], use_count

    # 反向传播
    def back(self, obj):
        # 获取父节点对象
        parent_one = self.all_node[obj.parent]
        # 去列表中找到这个对象并给对象中的value赋值
        while True:
            if parent_one.i == 0:
                break
            parent_one.c = parent_one.c + 1
            parent_one.value = ((obj.play * obj.value) + parent_one.value) / 2
            obj = parent_one
            parent_one = self.all_node[parent_one.parent]

    # 计算置信上限最大值
    def uct_max(self, policy, child):
        jz = {}
        policy = copy.deepcopy(policy)
        # 若此节点下棋局已经结束则无动作
        if len(policy) <= 0:
            print("棋局结束，无动作 ", end='')
            return False

        # 寻找最大值
        # 存储最大值
        max_value = -999
        # 存储最大值所在的key
        max_object = ''
        for k in list(policy.keys()):

            # 判断此动作是否在子节点中，若在则使用子节点的value进行UCT值计算
            if k in child:
                child_obj = self.all_node[child[k]]
                uct = child_obj.value + (5 * child_obj.policy_my) * (self.count_mcts_already / 1 + child_obj.c)
            else:
                uct = 5 * policy[k] * self.count_mcts_already
            jz[k] = uct
            # 比较大小
            if uct > max_value:
                max_value = uct
                max_object = k
        # print("总价值：{}".format(jz))
        # print("最大值：{}：{}".format(max_object,max_value))
        return max_object

    # 获取子节点字典终被选中次数最多的动作
    def final_max_count(self, all_obj):
        # 遍历最近四步棋，比较是否与policy动作存在两次冲突，存在则取消此动作，若取消后无动作，则判和返回：0
        policy_list = list(all_obj.child.keys())

        for y_x in policy_list:
            if self.y_x_list.count(y_x) >= 2:
                all_obj.child.pop(y_x)

        if len(all_obj.child) <= 0:
            return [0, 0], [all_obj.play]
        count = 0
        all_count = 0
        list_count = {}
        max_c = 0
        obj = ''
        for i in all_obj.child.keys():
            c = self.all_node[all_obj.child[i]].c
            list_count[i] = c
            count += c
            all_count += 1
            if c > max_c:
                obj = i
                max_c = c
        return [obj, max_c], list_count


# 网络预测结果
def net_result(data, play, y_x, z):
    tensor = get_date_one(data, play, y_x)
    # 对新节点进行神经网络预测，获取价值以及动作
    p1, v1 = use_zou(z, tensor)
    # 将策略的值转换为概率exp()函数
    p1 = exp(p1)
    # 筛选合法落子
    # 将p由张量格式转换为列表，将p与全部合法落子结合形成字典
    policy1 = dict(zip(piece_moves_all, p1.tolist()[0]))
    policy2 = {}
    # 生成当前局面的全部合法落子
    moves_all = legal_all_chess(data, play)
    # 通过以上两者最终确定合法落子与其概率
    for g1 in moves_all:
        policy2[g1] = policy1[g1]
    return policy2, v1


# 生成神经网络需要预测的数据
def get_date_one(situation, play, y_x):

    # 存储数据
    tensor = torch.zeros((1, 9, 10, 9))
    # 先将局面转为7通道张量
    tensor[0] = convert_chess_tensor(situation)
    # 添加对手上一步棋的位置
    if y_x == '0000':
        # 表示没有上一步，则全部设置为0
        tensor[0][7] = torch.zeros((10, 9))
    else:
        # 表示上一步有落子，则取出，并进行赋值
        np1 = torch.zeros((10, 9))
        np1[int(y_x[0])][int(y_x[1])] = -1  # 起始
        np1[int(y_x[2])][int(y_x[3])] = 1  # 结束
        tensor[0][7] = np1
    # 添加当前是谁下
    if play == '1':
        tensor[0][8] = torch.ones((10, 9))
    else:
        tensor[0][8] = -1 * torch.ones((10, 9))
    return tensor

# state_list_init = [['一一', '一一', '一一', '红帅', '一一', '红士', '红象', '红马', '红车'],
# ['红车', '一一', '一一', '一一', '红士', '一一', '一一', '一一', '一一'],
# ['红马', '一一', '一一', '一一', '红象', '一一', '一一', '一一', '红炮'],
# ['一一', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵'],
# ['黑兵', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
# ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '黑兵'],
# ['一一', '红炮', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '一一'],
# ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '黑象'],
# ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
# ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '一一', '黑马', '黑车']]
# # # state_list_init = [['红车', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
# # #                    ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
# # #                    ['一一', '红炮', '一一', '一一', '一一', '一一', '一一', '红炮', '一一'],
# # #                    ['红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵'],
# # #                    ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
# # #                    ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
# # #                    ['黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
# # #                    ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '一一'],
# # #                    ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
# # #                    ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑马', '黑车']]
# a = MCTSPlus(state_list_init, -1, 50, ['4041','4041'])
# a.mcts_train()
