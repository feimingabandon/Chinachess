# 开始自我对弈然后自我训练
# 1、准备初始化的棋盘
import copy
import tkinter
from datetime import datetime

state_list_init = [['红车', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '红炮', '一一', '一一', '一一', '一一', '一一', '红炮', '一一'],
                   ['红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
                   ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '一一'],
                   ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
                   ['黑车', '黑马', '黑象', '黑士', '黑帅', '黑士', '黑象', '黑马', '黑车']]
# 准备落子方、默认红方
play = 1
# 存储所有局面
all_chess = []
# 存储对手上一步棋，当前是谁下，当前局面下由蒙特卡洛得到的全部落子概率、以及落子方最终胜利情况:[['0000', '1', a, 1.0], ['1242', '-1', a, -1.0]]
state_list_init2 = []
# 存储最近八步棋
move_four = []

from mcts_puls import *
from game import *
from train import *


def mcts_p(mcts_policys):
    lst1 = [0] * 2086
    zd = dict(zip(copy.deepcopy(piece_moves_all), lst1))
    # 将通过蒙特卡洛搜索得到的概率传入字典
    for i1 in mcts_policys:
        zd[i1] = mcts_policys[i1]
    # 取字典的value重新输出列表
    return list(zd.values())


def move_four_s(move_fours, actions):
    actions2 = actions[2] + actions[3] + actions[0] + actions[1]
    if len(move_fours) <= 16:
        move_fours.append(actions)
        move_fours.append(actions2)
    else:
        move_fours.insert(0, actions)
        move_fours.insert(0, actions2)
        # 删除末尾元素
        move_fours.pop()
        move_fours.pop()
    return move_fours


def out_game_gui(state_init1, y_x):
    state_init1 = copy.deepcopy(state_init1)
    state_init1[int(y_x[0])][int(y_x[1])] = '▓▓▓'
    state_init1[int(y_x[2])][int(y_x[3])] = '▌' + str(state_init1[int(y_x[2])][int(y_x[3])])
    for s in state_init1:
        print(s)


for i in range(10):
    # 初始化数据
    state_list_init_ = copy.deepcopy(state_list_init)
    # 准备落子方、默认红方
    play_ = 1
    # 存储所有局面
    all_chess_ = []
    # 存储对手上一步棋，当前是谁下，当前局面下由蒙特卡洛得到的全部落子概率、以及落子方最终胜利情况:[['0000', '1', a, 1.0], ['1242', '-1', a, -1.0]]
    state_list_init2_ = []
    # 存储对方上一步落子
    y_x = '0000'
    # 开始自我对弈
    print("=================开始自我对弈 {} 次==============".format(i))
    # 记录当前时间
    from_time = datetime.now()
    # 使用神经网络进行预测
    z = zou()
    while True:
        # 查看一次蒙特卡洛树搜索时间
        from_time_2 = datetime.now()
        # 红方使用蒙特卡洛树搜索选择动作
        red_ = MCTSPlus(state_list_init_, play_, count_mcts=300, y_x_list=move_four, z=z)
        action, mcts_policy = red_.mcts_train()
        if action[0] == action[1] == 0:
            print("游戏结束,败方：{}".format(mcts_policy[0]))
            # 将数据集最后一部分填满
            for j in range(len(state_list_init2_)):
                if state_list_init2_[j][1] == mcts_policy[0]:
                    state_list_init2_[j][3] = mcts_policy[0]
                else:
                    state_list_init2_[j][3] = -mcts_policy[0]
            break
        move_four = move_four_s(move_four, action)
        print(move_four)
        to_time_2 = datetime.now()
        print("此次蒙特卡洛树搜索使用时间：{} ,单位（秒）".format((to_time_2-from_time_2).seconds))
        # 根据蒙特卡洛树搜索的结果进行移动局面
        move_chess(state_list_init_, action, legal_all_chess(state_list_init_, play_))
        # 收集对弈数据
        all_chess_.append(copy.deepcopy(state_list_init_))
        # 生成蒙特卡洛搜索的概率分布
        state_list_init2_.append([y_x, play_, mcts_p(mcts_policy), ''])
        # 反转play
        play_ = -play_
        print("{} 方下，走子为：{},局面为：".format(-play_, action))

        out_game_gui(state_list_init_, action)
        # 存储此次走子
        y_x = action
        # 判断是否胜利
        win_ = win_fill_draw(state_list_init_)
        if win_ != 0:
            # 将数据集最后一部分填满
            for j in range(len(state_list_init2_)):
                if state_list_init2_[j][1] == win_:
                    if 50 >= len(state_list_init2_):
                        state_list_init2_[j][3] = win_
                    elif 100 > len(state_list_init2_) > 50:
                        state_list_init2_[j][3] = win_ - 0.25
                    elif len(state_list_init2_) > 100:
                        state_list_init2_[j][3] = win_ - 0.5

                else:
                    state_list_init2_[j][3] = -win_
            if win_ == 1:
                print("红胜")
            else:
                print("黑胜")
            break
        print("步数：{}".format(len(all_chess_)))
    to_time = datetime.now()
    print("此次自我对弈总用时：{} ,单位（分）".format((to_time - from_time).seconds / 60))
    # 获取数据、进行训练
    print("=============开始训练=================")
    train_zou_net(all_chess_, state_list_init2_)



