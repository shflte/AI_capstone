import STcpClient
import numpy as np
import random

'''
    產出指令
    
    input: 
    playerID: 你在此局遊戲中的角色(1~4)
    mapStat : 棋盤狀態(list of list), 為 12*12矩陣, 
              0=可移動區域, -1=障礙, 1~4為玩家1~4佔領區域
    sheepStat : 羊群分布狀態, 範圍在0~16, 為 12*12矩陣

    return Step
    Step : 3 elements, [(x,y), m, dir]
            x, y 表示要進行動作的座標 
            m = 要切割成第二群的羊群數量
            dir = 移動方向(1~9),對應方向如下圖所示
            1 2 3
            4 X 6
            7 8 9
'''
# implement the step function with MCTS

class GameInteraction:
    # possible directions in the game
    def possible_dir(self):
        return [1, 2, 3, 4, 6, 7, 8, 9]

    # value of each direction
    def dir_value(self, dir):
        if dir == 1:
            return (-1, -1)
        elif dir == 2:
            return (-1, 0)
        elif dir == 3:
            return (-1, 1)
        elif dir == 4:
            return (0, -1)
        elif dir == 5:
            return (0, 0)
        elif dir == 6:
            return (0, 1)
        elif dir == 7:
            return (1, -1)
        elif dir == 8:
            return (1, 0)
        elif dir == 9:
            return (1, 1)

    # probe the direction until out of the map or hit something that is not 0
    def probe_direction(self, x, y, dir, mapStat):
        dx, dy = self.dir_value(dir)
        while True:
            if not (0 <= x + dx < len(mapStat) and 0 <= y + dy < len(mapStat[0])):
                break
            if mapStat[x + dx][y + dy] != 0:
                break
            x += dx
            y += dy
        return (x, y)

    # return new state after applying action
    def apply_action(self, state, action):
        playerID, mapStat, sheepStat = state
        newMapStat = mapStat.copy()
        newSheepStat = sheepStat.copy()
        
        x, y = action[0]
        m = action[1]
        dir = action[2]

        newx, newy = self.probe_direction(x, y, dir, mapStat)
        # update newMapStat
        newMapStat[newx][newy] = playerID
        # update newSheepStat
        newSheepStat[x][y] -= m
        newSheepStat[newx][newy] = m

        return (playerID, newMapStat, newSheepStat)

    # return all possible actions
    def get_possible_actions(self, state):
        playerID, mapStat, sheepStat = state
        actions = []
        height, width = len(mapStat), len(mapStat[0])
        # find all splitable sheep group and append the groups' index to splitable_sheep
        splitable_sheep = []
        for i in range(height):
            for j in range(width):
                if mapStat[i][j] == playerID and sheepStat[i][j] > 1:
                    splitable_sheep.append((i, j))
        # for each splitable sheep group, generate all possible actions
        for i, j in splitable_sheep:
            for m in range(1, sheepStat[i][j]):
                for dir in self.possible_dir():
                    newx, newy = self.probe_direction(i, j, dir, mapStat)
                    if newx != i or newy != j:
                        actions.append([(i, j), m, dir])

        return actions

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0

class Tree:
    def __init__(self, state):
        self.root = Node(state)

class MCTS:
    def __init__(self, state):
        self.tree = Tree(state)

    def select(self, node):
        while node.children:
            node = max(node.children, key=lambda n: self.ucb(n))
        return node

    def expand(self, node):
        actions = self.get_possible_actions(node.state)
        for action in actions:
            new_state = self.apply_action(node.state, action)
            new_node = Node(new_state, node)
            node.children.append(new_node)
        return random.choice(node.children)

    def simulate(self, state):

        return random.random()

    def backpropagate(self, node, value):
        while node:
            node.visits += 1
            node.value += value
            node = node.parent

    def ucb(self, node):
        if node.visits == 0:
            return float('inf')
        return node.value / node.visits + (2 * np.log(node.parent.visits) / node.visits) ** 0.5

    def get_action(self):
        root = self.tree.root
        for _ in range(1000):
            selected_node = self.select(root)
            if not selected_node.children:
                expanded_node = self.expand(selected_node)
                value = self.simulate(expanded_node.state)
                self.backpropagate(expanded_node, value)
            else:
                value = self.simulate(selected_node.state)
                self.backpropagate(selected_node, value)
        best_child = max(root.children, key=lambda n: n.visits)
        return self.get_best_action(root, best_child)

    def get_best_action(self, root, best_child):
        # 这里根据实际需要返回最佳动作
        return None

    def get_possible_actions(self, state):
        # 根据当前状态返回所有可能的动作
        return []
    
    def apply_action(self, state, action):
        

        return state

'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置
    
'''
def InitPos(mapStat):
    init_pos = [0, 0]
    height, width = len(mapStat), len(mapStat[0])
    
    # randomly choose a position on the edge of the map
    for i in range(height):
        if mapStat[i][0] == -1:
            init_pos = [i, 0]
            break
        elif mapStat[i][width - 1] == -1:
            init_pos = [i, width - 1]
            break
    for j in range(width):
        if mapStat[0][j] == -1:
            init_pos = [0, j]
            break
        elif mapStat[height - 1][j] == -1:
            init_pos = [height - 1, j]
            break
    
    return init_pos

def GetStep(playerID, mapStat, sheepStat):
    mcts = MCTS((playerID, mapStat, sheepStat))
    best_action = mcts.get_action()
    step = [(best_action[0], best_action[1]), best_action[2], best_action[3]]

    return step


# player initial
# (id_package, playerID, mapStat) = STcpClient.GetMap()
# init_pos = InitPos(mapStat)
# STcpClient.SendInitPos(id_package, init_pos)

# start game
# while (True):
#     (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
#     if end_program:
#         STcpClient._StopConnect()
#         break
#     Step = GetStep(playerID, mapStat, sheepStat)

#     STcpClient.SendStep(id_package, Step)


# generate a toy 12 * 12 toy mapStat
mapStat = [[0 for _ in range(12)] for _ in range(12)]
# generate a toy 12 * 12 toy sheepStat
sheepStat = [[0 for _ in range(12)] for _ in range(12)]

# player 2 has 10 sheep at (1, 1), 6 sheep at (3, 3)
mapStat[0][0] = 2
sheepStat[0][0] = 2
mapStat[1][0] = 2
sheepStat[1][0] = 1
mapStat[0][1] = 2
sheepStat[0][1] = 1

init_pos = InitPos(mapStat)
all_actions  = GameInteraction().get_possible_actions((2, mapStat, sheepStat))
print(all_actions)