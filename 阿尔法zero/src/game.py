# 生成象棋游戏
import torch

from ChessData import state_init

class game():
    def __init__(self):
        # 存储当前是哪一方下棋
        self.game_black_red = 1  # 默认红方先行

        # 初始化棋盘
        #self.state_init1 = state_init

        # 定义一个列表存储最近4步棋,删除末尾数据pop方法，头部添加数据insert(0, "嘿嘿嘿")
        self.chess_four = []

        # 定义一个列表存储每个局面
        self.state_list_init = []

        # 1、存储对手上一步棋，2、当前是谁下，3、当前局面下由蒙特卡洛得到的全部落子概率、4、以及落子方最终胜利情况
        self.state_list_init2 = []


# 通过坐标移动棋子(或者吃子),注：x,y需要交换位置，因为正常情况下是[y][x]作为二维数组定位到指定值，但是我们这里使用的是x,y所有x_y变量里面的位置需要换一下
def move_chess(state_init1 , y_x, moves_all):
    # 移动前先判断是否合法
    if legal_chess(y_x, moves_all):
        # 存储要移动的棋子
        chess = state_init1[int(y_x[0])][int(y_x[1])]
        # 移动
        # 将原点设置为空
        state_init1[int(y_x[0])][int(y_x[1])] = '一一'
        # 将终点设置为需要移动的棋子
        state_init1[int(y_x[2])][int(y_x[3])] = chess
    return state_init1


# 以当前局面判断落子是否合法，前提是初始位置上是棋子且是落子方的棋子
# game_black_red:哪一方下，y_x:一个字符串变量，存储四个数，表示起始坐标与终止坐标，如(0001)
def legal_chess( y_x, moves_all):
    if y_x in moves_all:
        return True
    return False


# 生成当前局面某一方的所有合法落子
def legal_all_chess(state_init1, game_black_red):
    # 存放当前局面下某一方的全部合法走子
    moves_all = []
    # 遍历棋盘上的所有位置
    for y in range(10):
        for x in range(9):
            # 表示是空的，不需要进行合法判断
            if state_init1[y][x] == '一一':
                continue
            elif game_black_red == 1:
                # 表示红方
                # 接下来判断红方棋子能够走的位置
                if state_init1[y][x] == '红帅':
                    # 将 能够向 上 下 左 右 方移动一步
                    # 获取全部终点坐标
                    y_x = [[y-1, x], [y+1, x], [y, x-1], [y, x+1]]
                    # 将合法的取出并存储进moves_all中
                    moves_all += boundary(y, x, y_x, [0, 3, 2, 5], '红', state_init1)
                    for i in range(9):
                        if y+i+1 <= 9 and state_init1[y+i+1][x] != '一一':
                            if state_init1[y+i+1][x] == '黑帅':
                                moves_all.append(str(y) + str(x) + str(y+i+1) + str(x))
                            break
                    continue
                if state_init1[y][x] == '红士':
                    # 士 能向 左下 右下 左上 右上移动
                    # 获取全部终点坐标
                    y_x = [[y+1, x-1], [y+1, x+1], [y-1, x-1], [y-1, x+1]]
                    # 将合法的取出并存储进moves_all中
                    moves_all += boundary(y, x, y_x, [0, 3, 2, 5], '红', state_init1)
                    continue
                if state_init1[y][x] == '红象':
                    # 象 能向 左下 右下 左上 右上 以田字移动(0,2->2,0)(0,2->2,4)(0,2->-2,0)(0,2->-2,4)
                    # 获取全部终点坐标
                    y_x = [[y + 2, x - 2], [y + 2, x + 2], [y - 2, x - 2], [y - 2, x + 2]]

                    # 使用循环判断是否合法
                    for drop in y_x:
                        # 判断是否超出边界
                        if 0 <= drop[0] <= 4 and 0 <= drop[1] <= 8:
                            # 表示不超出
                            # 判断终点坐标上是否有子
                            if state_init1[drop[0]][drop[1]] == '一一' or (state_init1[drop[0]][drop[1]])[0] != '红':
                                # 判断田字中间是否有子，有则不行
                                if state_init1[int((y+drop[0])/2)][int((x+drop[1])/2)] == '一一':
                                    # 终点不是己方子就行
                                    moves_all.append(str(y) + str(x) + str(drop[0]) + str(drop[1]))
                    continue
                if state_init1[y][x] == '红马':
                    # 马能向 左下 右下 左上 右上 移动8个方位(3,4 -> 1,3)(3,4 -> 2,2)(3,4 -> 1,5)(3,4 -> 2,6)(3,4 -> 4,2)(3,4 -> 5,3)(3,4 -> 4,6)(3,4 -> 5,5)
                    y_x = []  # [[y-2, x-1], [y-1, x-2], [y-2, x+1], [y-1, x+2], [y+1, x-2], [y+2,x-1], [y+1, x+2], [y+2, x+1]]
                    # 筛选出被蹩脚的终点,蹩脚只需要查看原点的上下左右是否存在棋子，如果存在则某一方向上就不能到达
                    if 0 <= y-1 <= 9 and state_init1[y-1][x] == '一一': # 上
                        y_x += [[y-2, x-1], [y-2, x+1]]
                    if 0 <= y+1 <= 9 and state_init1[y+1][x] == '一一': # 下
                        y_x += [[y+2, x-1], [y+2, x+1]]
                    if 0 <= x-1 <= 8 and state_init1[y][x-1] == '一一': # 左
                        y_x += [[y-1, x-2], [y+1, x-2]]
                    if 0 <= x+1 <= 8 and state_init1[y][x+1] == '一一': # 右
                        y_x += [[y-1, x+2], [y+1, x+2]]

                    # 取合法坐标
                    moves_all += boundary(y, x, y_x, [0, 0, 9, 8], '红', state_init1)
                    continue
                if state_init1[y][x] == '红车':
                    # 上下左右 横移
                    # 计算能够横移的全部坐标
                    # 计算上方向的合法落子
                    for i in range(9):
                        if 0 <= y+(i+1) <= 9 and state_init1[y+(i+1)][x] == '一一':
                            moves_all.append(str(y) + str(x) + str(y+(i+1)) + str(x))
                        else:
                            if y+(i+1) < 0 or y+(i+1) > 9:
                                break
                            elif (state_init1[y+(i+1)][x])[0] == '黑':
                                # 表示可以吃
                                moves_all.append(str(y) + str(x) + str(y+(i+1)) + str(x))
                                break
                            elif (state_init1[y+(i+1)][x])[0] == '红':
                                break

                    # 计算下方向的合法落子
                    for i in range(9):
                        if  0 <= y-(i+1) <= 9 and state_init1[y-(i+1)][x] == '一一':
                            moves_all.append(str(y) + str(x) + str(y-(i+1)) + str(x))
                        else:
                            if y-(i+1) < 0 or y-(i+1) > 9:
                                break
                            elif (state_init1[y-(i+1)][x])[0] == '黑':
                                # 表示可以吃
                                moves_all.append(str(y) + str(x) + str(y-(i+1)) + str(x))
                                break
                            elif (state_init1[y-(i+1)][x])[0] == '红':
                                break
                    # 计算左方向的合法落子
                    for i in range(8):
                        if 0 <= x+(i+1) <= 8 and state_init1[y][x+(i+1)] == '一一':
                            moves_all.append(str(y) + str(x) + str(y) + str(x+(i+1)))
                        else:
                            if x+(i+1) < 0 or x+(i+1) > 8:
                                break
                            elif (state_init1[y][x+(i+1)])[0] == '黑':
                                # 表示可以吃
                                moves_all.append(str(y) + str(x) + str(y) + str(x+(i+1)))
                                break
                            elif (state_init1[y][x+(i+1)])[0] == '红':
                                break
                    # 计算右方向的合法落子
                    for i in range(8):
                        if  0 <= x-(i+1) <= 8 and state_init1[y][x-(i+1)] == '一一':
                            moves_all.append(str(y) + str(x) + str(y) + str(x-(i+1)))
                        else:
                            if x-(i+1) < 0 or x-(i+1) > 8:
                                break
                            elif (state_init1[y][x-(i+1)])[0] == '黑':
                                # 表示可以吃
                                moves_all.append(str(y) + str(x) + str(y) + str(x-(i+1)))
                                break
                            elif (state_init1[y][x-(i+1)])[0] == '红':
                                break

                    continue
                if state_init1[y][x] == '红炮':
                    # 上下左右横移，如果有阻碍且阻碍后是对方的棋子，则可以打吃
                    # 和车的走法差不多，但是遇到阻碍时会有点不一样
                    for i in range(9):
                        flag = False
                        if 0 <= y+(i+1) <= 9 and state_init1[y+(i+1)][x] == '一一':
                            moves_all.append(str(y) + str(x) + str(y+(i+1)) + str(x))
                        else:
                            flag = True
                            if y+(i+1) < 0 or y+(i+1) > 9:
                                break
                            else:
                                # 查看此子同方向后面是否有对方的子，若有则添加
                                for j in range(9):
                                    if 0 <= y+(i+1)+(j+1) <= 9:
                                        if (state_init1[y+(i+1)+(j+1)][x])[0] == '黑':
                                            # 表示有，则可以打吃
                                            moves_all.append(str(y) + str(x) + str(y+(i+1)+(j+1)) + str(x))
                                            # 然后退出

                                            break
                                    else:
                                        break
                        if flag:
                            break
                    # 计算下方向的合法落子
                    for i in range(9):
                        flag = False
                        if  0 <= y-(i+1) <= 9 and state_init1[y-(i+1)][x] == '一一':
                            moves_all.append(str(y) + str(x) + str(y-(i+1)) + str(x))
                        else:
                            flag = True
                            if y-(i+1) < 0 or y-(i+1) > 9:
                                break
                            else:
                                # 查看此子同方向后面是否有对方的子，若有则添加
                                for j in range(9):
                                    if 0 <= y-(i+1)-(j+1) <= 9:
                                        if (state_init1[y-(i+1)-(j+1)][x])[0] == '黑':
                                            # 表示有，则可以打吃
                                            moves_all.append(str(y) + str(x) + str(y-(i+1)-(j+1)) + str(x))
                                            # 然后退出

                                            break
                                    else:
                                        break
                        if flag:
                            break
                    # 计算左方向的合法落子
                    for i in range(8):
                        flag = False
                        if  0 <= x+(i+1) <= 8 and state_init1[y][x+(i+1)] == '一一':
                            moves_all.append(str(y) + str(x) + str(y) + str(x+(i+1)))
                        else:
                            flag = True
                            if x+(i+1) < 0 or x+(i+1) > 8:
                                break
                            else:
                                # 查看此子同方向后面是否有对方的子，若有则添加
                                for j in range(8):
                                    if 0 <= x+(i+1)+(j+1) <= 8:
                                        if (state_init1[y][x+(i+1)+(j+1)])[0] == '黑':
                                            # 表示有，则可以打吃
                                            moves_all.append(str(y) + str(x) + str(y) + str(x+(i+1)+(j+1)))
                                            # 然后退出

                                            break
                                    else:
                                        break
                        if flag:
                            break
                    # 计算右方向的合法落子
                    for i in range(8):
                        flag = False
                        if  0 <= x-(i+1) <= 8 and state_init1[y][x-(i+1)] == '一一':
                            moves_all.append(str(y) + str(x) + str(y) + str(x-(i+1)))
                        else:
                            flag = True
                            if x-(i+1) < 0 or x-(i+1) > 8:
                                break
                            else:
                                # 查看此子同方向后面是否有对方的子，若有则添加
                                for j in range(9):
                                    if 0 <= x-(i+1)-(j+1) <= 8:
                                        if (state_init1[y][x-(i+1)-(j+1)])[0] == '黑':
                                            # 表示有，则可以打吃
                                            moves_all.append(str(y) + str(x) + str(y) + str(x-(i+1)-(j+1)))
                                            # 然后退出

                                            break
                                    else:
                                        break
                        if flag:
                            break
                    continue
                if state_init1[y][x] == '红兵':
                    # 过河道后可以左右上移，没过只能上移
                    # 如果是没过河
                    if y <= 4:
                        # 只能向前走一步，且前方没有我方的子
                        if state_init1[y+1][x] == '一一' or (state_init1[y+1][x])[0] == '黑':
                            moves_all.append(str(y) + str(x) + str(y+1) + str(x))
                    elif 9 >= y > 4:
                        # 可以前进左右，就是不能后退
                        # 前进
                        if 9 >= y+1 > 4:
                            if state_init1[y+1][x] == '一一' or (state_init1[y+1][x])[0] == '黑':
                                moves_all.append(str(y) + str(x) + str(y + 1) + str(x))
                        # 左边
                        if 0 <= x-1 <= 8:
                            if state_init1[y][x-1] == '一一' or (state_init1[y][x-1])[0] == '黑':
                                moves_all.append(str(y) + str(x) + str(y) + str(x-1))
                        # 右边
                        # 左边
                        if 0 <= x+1 <= 8:
                            if state_init1[y][x+1] == '一一' or (state_init1[y][x+1])[0] == '黑':
                                moves_all.append(str(y) + str(x) + str(y) + str(x+1))
                continue

            elif game_black_red == -1:
                # 表示黑方
                # 接下来判断黑方棋子能够走的位置
                if state_init1[y][x] == '黑帅':
                    # 将 能够向 上 下 左 右 方移动一步
                    # 获取全部终点坐标
                    y_x = [[y - 1, x], [y + 1, x], [y, x - 1], [y, x + 1]]
                    # 将合法的取出并存储进moves_all中
                    moves_all += boundary(y, x, y_x, [7, 3, 9, 5], '黑', state_init1)
                    for i in range(9):
                        if y-(i+1) >= 0 and state_init1[y-(i+1)][x] != '一一':
                            if state_init1[y-(i+1)][x] == '红帅':
                                moves_all.append(str(y) + str(x) + str(y-(i+1)) + str(x))
                            break
                    continue
                if state_init1[y][x] == '黑士':
                    # 士 能向 左下 右下 左上 右上移动
                    # 获取全部终点坐标
                    y_x = [[y + 1, x - 1], [y + 1, x + 1], [y - 1, x - 1], [y - 1, x + 1]]
                    # 将合法的取出并存储进moves_all中
                    moves_all += boundary(y, x, y_x, [7, 3, 9, 5], '黑', state_init1)
                    continue
                if state_init1[y][x] == '黑象':
                    # 象 能向 左下 右下 左上 右上 以田字移动(0,2->2,0)(0,2->2,4)(0,2->-2,0)(0,2->-2,4)
                    # 获取全部终点坐标
                    y_x = [[y + 2, x - 2], [y + 2, x + 2], [y - 2, x - 2], [y - 2, x + 2]]
                    # 使用循环判断是否合法
                    for drop in y_x:
                        # 判断是否超出一方边界(0,3)(2,5)(y,x)
                        if 5 <= drop[0] <= 9 and 0 <= drop[1] <= 8:
                            # 表示不超出
                            # 判断终点坐标上是否有子
                            if state_init1[drop[0]][drop[1]] == '一一' or (state_init1[drop[0]][drop[1]])[0] != '黑':
                                # 判断田字中间是否有子，有则不行
                                if state_init1[int((y+drop[0])/2)][int((x+drop[1])/2)] == '一一':
                                    # 终点不是己方子就行
                                    moves_all.append(str(y) + str(x) + str(drop[0]) + str(drop[1]))
                    continue
                if state_init1[y][x] == '黑马':
                    # 马能向 左下 右下 左上 右上 移动8个方位(3,4 -> 1,3)(3,4 -> 2,2)(3,4 -> 1,5)(3,4 -> 2,6)(3,4 -> 4,2)(3,4 -> 5,3)(3,4 -> 4,6)(3,4 -> 5,5)
                    y_x = []  # [[y-2, x-1], [y-1, x-2], [y-2, x+1], [y-1, x+2], [y+1, x-2], [y+2,x-1], [y+1, x+2], [y+2, x+1]]
                    # 筛选出被蹩脚的终点,蹩脚只需要查看原点的上下左右是否存在棋子，如果存在则某一方向上就不能到达
                    if 0 <= y - 1 <= 9 and state_init1[y - 1][x] == '一一':  # 上
                        y_x += [[y - 2, x - 1], [y - 2, x + 1]]
                    if 0 <= y + 1 <= 9 and state_init1[y + 1][x] == '一一':  # 下
                        y_x += [[y + 2, x - 1], [y + 2, x + 1]]
                    if 0 <= x - 1 <= 8 and state_init1[y][x - 1] == '一一':  # 左
                        y_x += [[y - 1, x - 2], [y + 1, x - 2]]
                    if 0 <= x + 1 <= 8 and state_init1[y][x + 1] == '一一':  # 右
                        y_x += [[y - 1, x + 2], [y + 1, x + 2]]

                    # 取合法坐标
                    moves_all += boundary(y, x, y_x, [0, 0, 9, 8], '黑', state_init1)
                    continue
                if state_init1[y][x] == '黑车':
                    # 上下左右 横移
                    # 计算能够横移的全部坐标
                    # 计算上方向的合法落子
                    for i in range(9):
                        if 0 <= y + (i + 1) <= 9 and state_init1[y + (i + 1)][x] == '一一':
                            moves_all.append(str(y) + str(x) + str(y + (i + 1)) + str(x))
                        else:
                            if y + (i + 1) < 0 or y + (i + 1) > 9:
                                break
                            elif (state_init1[y + (i + 1)][x])[0] == '红':
                                # 表示可以吃
                                moves_all.append(str(y) + str(x) + str(y + (i + 1)) + str(x))
                                break
                            elif (state_init1[y+(i+1)][x])[0] == '黑':
                                break
                    # 计算下方向的合法落子
                    for i in range(9):
                        if 0 <= y - (i + 1) <= 9 and state_init1[y - (i + 1)][x] == '一一':
                            moves_all.append(str(y) + str(x) + str(y - (i + 1)) + str(x))
                        else:
                            if y - (i + 1) < 0 or y - (i + 1) > 9:
                                break
                            elif (state_init1[y - (i + 1)][x])[0] == '红':
                                # 表示可以吃
                                moves_all.append(str(y) + str(x) + str(y - (i + 1)) + str(x))
                                break
                            elif (state_init1[y - (i + 1)][x])[0] == '黑':
                                break
                    # 计算左方向的合法落子
                    for i in range(8):
                        if 0 <= x + (i + 1) <= 8 and state_init1[y][x + (i + 1)] == '一一':
                            moves_all.append(str(y) + str(x) + str(y) + str(x + (i + 1)))
                        else:
                            if x + (i + 1) < 0 or x + (i + 1) > 8:
                                break
                            elif (state_init1[y][x + (i + 1)])[0] == '红':
                                # 表示可以吃
                                moves_all.append(str(y) + str(x) + str(y) + str(x + (i + 1)))
                                break
                            elif (state_init1[y][x + (i + 1)])[0] == '黑':
                                break
                    # 计算右方向的合法落子
                    for i in range(8):
                        if 0 <= x - (i + 1) <= 8 and state_init1[y][x - (i + 1)] == '一一':
                            moves_all.append(str(y) + str(x) + str(y) + str(x - (i + 1)))
                        else:
                            if x - (i + 1) < 0 or x - (i + 1) > 8:
                                break
                            elif (state_init1[y][x - (i + 1)])[0] == '红':
                                # 表示可以吃
                                moves_all.append(str(y) + str(x) + str(y) + str(x - (i + 1)))
                                break
                            elif (state_init1[y][x - (i + 1)])[0] == '黑':
                                break

                    continue
                if state_init1[y][x] == '黑炮':
                    # 上下左右横移，如果有阻碍且阻碍后是对方的棋子，则可以打吃
                    # 和车的走法差不多，但是遇到阻碍时会有点不一样
                    for i in range(9):
                        flag = False
                        if 0 <= y + (i + 1) <= 9 and state_init1[y + (i + 1)][x] == '一一':
                            moves_all.append(str(y) + str(x) + str(y + (i + 1)) + str(x))
                        else:
                            flag = True
                            if y + (i + 1) < 0 or y + (i + 1) > 9:
                                break
                            else:
                                # 查看此子同方向后面是否有对方的子，若有则添加
                                for j in range(9):
                                    if 0 <= y + (i + 1) + (j + 1) <= 9:
                                        if (state_init1[y + (i + 1) + (j + 1)][x])[0] == '红':
                                            # 表示有，则可以打吃
                                            moves_all.append(str(y) + str(x) + str(y + (i + 1) + (j + 1)) + str(x))
                                            # 然后退出
                                            break
                                    else:
                                        break

                        if flag:
                            break
                    # 计算下方向的合法落子
                    for i in range(9):
                        flag = False
                        if 0 <= y - (i + 1) <= 9 and state_init1[y - (i + 1)][x] == '一一':
                            moves_all.append(str(y) + str(x) + str(y - (i + 1)) + str(x))
                        else:
                            flag = True
                            if y - (i + 1) < 0 or y - (i + 1) > 9:
                                break
                            else:
                                # 查看此子同方向后面是否有对方的子，若有则添加
                                for j in range(9):
                                    if 0 <= y - (i + 1) - (j + 1) <= 9:
                                        if (state_init1[y - (i + 1) - (j + 1)][x])[0] == '红':
                                            # 表示有，则可以打吃
                                            moves_all.append(str(y) + str(x) + str(y - (i + 1) - (j + 1)) + str(x))
                                            # 然后退出

                                            break
                                    else:
                                        break
                        if flag:
                            break
                    # 计算左方向的合法落子
                    for i in range(8):
                        flag = False
                        if 0 <= x + (i + 1) <= 8 and state_init1[y][x + (i + 1)] == '一一':
                            moves_all.append(str(y) + str(x) + str(y) + str(x + (i + 1)))
                        else:
                            flag = True
                            if x + (i + 1) < 0 or x + (i + 1) > 8:
                                break
                            else:
                                # 查看此子同方向后面是否有对方的子，若有则添加
                                for j in range(8):
                                    if 0 <= x + (i + 1) + (j + 1) <= 8:
                                        if (state_init1[y][x + (i + 1) + (j + 1)])[0] == '红':
                                            # 表示有，则可以打吃
                                            moves_all.append(str(y) + str(x) + str(y) + str(x + (i + 1) + (j + 1)))
                                            # 然后退出

                                            break
                                    else:
                                        break
                        if flag:
                            break
                    # 计算右方向的合法落子
                    for i in range(8):
                        flag = False
                        if 0 <= x - (i + 1) <= 8 and state_init1[y][x - (i + 1)] == '一一':
                            moves_all.append(str(y) + str(x) + str(y) + str(x - (i + 1)))
                        else:
                            flag = True
                            if x - (i + 1) < 0 or x - (i + 1) > 8:
                                break
                            else:
                                # 查看此子同方向后面是否有对方的子，若有则添加
                                for j in range(9):
                                    if 0 <= x - (i + 1) - (j + 1) <= 8:
                                        if (state_init1[y][x - (i + 1) - (j + 1)])[0] == '红':
                                            # 表示有，则可以打吃
                                            moves_all.append(str(y) + str(x) + str(y) + str(x - (i + 1) - (j + 1)))
                                            # 然后退出

                                            break
                                    else:
                                        break
                        if flag:
                            break
                    continue
                if state_init1[y][x] == '黑兵':
                    # 过河道后可以左右上移，没过只能上移
                    # 如果是没过河
                    if 9 >= y > 4:
                        # 只能向前走一步，且前方没有我方的子
                        if state_init1[y - 1][x] == '一一' or (state_init1[y - 1][x])[0] == '红':
                            moves_all.append(str(y) + str(x) + str(y - 1) + str(x))
                    elif 0 <= y <= 4:
                        # 可以前进左右，就是不能后退
                        # 前进
                        if 4 >= y + 1 >= 0:
                            if state_init1[y - 1][x] == '一一' or (state_init1[y - 1][x])[0] == '红':
                                moves_all.append(str(y) + str(x) + str(y - 1) + str(x))
                        # 左边
                        if 0 <= x - 1 <= 8:
                            if state_init1[y][x - 1] == '一一' or (state_init1[y][x - 1])[0] == '红':
                                moves_all.append(str(y) + str(x) + str(y) + str(x - 1))
                        # 右边
                        if 0 <= x + 1 <= 8:
                            if state_init1[y][x + 1] == '一一' or (state_init1[y][x + 1])[0] == '红':
                                moves_all.append(str(y) + str(x) + str(y) + str(x + 1))
                continue
    return moves_all


# 边界检测，以及合法检测，正常情况下落子位置不能小于(0,0)，不能大于(8,9)，但是将，士，象不一样
# y_x ：存储所有能够走的坐标，不考虑合法，boundary_y_x：存储四个变量，表示当前类型的子的走子边界,按照范围的左上角与右下角填写, y, x:原点坐标
def boundary(y, x, y_x,boundary_y_x, color, state_init2):
    # 存储合法走子
    moves_all = []
    # 使用循环判断是否合法
    for drop in y_x:
        # 判断是否超出田字格边界(0,3)(2,5)(y,x)
        if boundary_y_x[0] <= drop[0] <= boundary_y_x[2] and boundary_y_x[1] <= drop[1] <= boundary_y_x[3]:
            # 表示不超出
            # 判断终点坐标上是否有子
            if state_init2[drop[0]][drop[1]] == '一一' or (state_init2[drop[0]][drop[1]])[0] != color:
                # 终点不是己方子就行
                moves_all.append(str(y) + str(x) + str(drop[0]) + str(drop[1]))
    return moves_all


# 判断是否胜利或者平局、判断将是否还在就行
def win_fill_draw(state_init1):
    red_win = 0
    black_win = 0
    for y in range(10):
        for x in range(9):
            # 循环遍历判断一方的帅是不是没了
            if state_init1[y][x] == '红帅':
                red_win = 1
            if state_init1[y][x] == '黑帅':
                black_win = 1
    if red_win == 0:
        return -1
    if black_win == 0:
        return 1

    return 0  # 1、-1：胜利，0：未分出胜负


# move_chess(state_init1, '2124')
# print(legal_all_chess(state_init1, 1))

# 输出局面
def out_game(state_init1):
    for s in state_init1:
        print(s)



    # # 开始游戏
    # def game_go(self):
    #     # 初始化局面
    #     self.state_init1 = state_init
    #     self.game_black_red = 1
    #     print("初始化局面：")
    #     self.out_game(self.state_init1)
    #     while True:
    #         # 红方选择落子
    #         # 输入局面与落子方，计算所有合法落子
    #         moves_all = self.legal_all_chess(self.state_init1, self.game_black_red)
    #         # 通过蒙特卡洛树搜索进行所有落子概率计算
    #         y_x = '0010'  # 假设最终落子坐标
    #         # 判断落子是否合法
    #         if self.legal_chess(y_x, moves_all):
    #             # 得到最终落子坐标、将坐标放入移动函数进行移动
    #             self.move_chess(self.state_init1, y_x, moves_all)
    #             # 判断是否胜利
    #             win = self.win_fill_draw(self.state_init1)
    #             # 输出局面
    #             print("落子方：{}，选择落子，{}".format(self.game_black_red, y_x))
    #             self.out_game(self.state_init1)
    #             # 换手
    #             if self.game_black_red == 1:
    #                 self.game_black_red = -1
    #             else:
    #                 self.game_black_red = 1
    #             print("该 {} 方落子".format(self.game_black_red))
    #             if win == 1:
    #                 print("红胜")
    #                 break
    #             elif win == -1:
    #                 print("黑胜")
    #                 break
    #         else:
    #             print("落子不合法，请重新选择")
    #         break
# s = [['一一', '红马', '红象', '红士', '红帅', '红士', '红象', '红马', '红车'],
#      ['红车', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
#      ['一一', '红炮', '一一', '一一', '一一', '一一', '一一', '红炮', '一一'],
#      ['红兵', '一一', '红兵', '一一', '红兵', '一一', '红兵', '一一', '一一'],
#      ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '红兵'],
#      ['一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一', '一一'],
#      ['黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵', '一一', '黑兵'],
#      ['一一', '黑炮', '一一', '一一', '一一', '一一', '一一', '黑炮', '黑象'],
#      ['一一', '一一', '一一', '一一', '黑帅', '一一', '一一', '一一', '一一'],
#      ['黑车', '黑马', '黑象', '黑士', '一一', '一一', '一一', '一一', '一一']]
# print(legal_all_chess(s, -1))