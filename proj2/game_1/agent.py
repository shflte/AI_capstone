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

class GameInteraction:
    # flip row and column of the game state (mapStat, sheepStat)
    def flip(self, state):
        playerID, mapStat, sheepStat = state
        newMapStat = mapStat.copy()
        newSheepStat = sheepStat.copy()
        for i in range(len(mapStat)):
            for j in range(len(mapStat[0])):
                newMapStat[j][i] = mapStat[i][j]
                newSheepStat[j][i] = sheepStat[i][j]
        return (playerID, newMapStat, newSheepStat)

    # flip direction of the action. x, y remain the same.
    def flip_action(self, action):
        dir_table = {1: 1, 2: 4, 3: 7, 4: 2, 6: 8, 7: 3, 8: 6, 9: 9}
        x, y = action[0]
        m = action[1]
        dir = action[2]
        newx, newy = x, y
        newdir = dir_table[dir]

        return [(newx, newy), m, newdir]

    # possible directions in the game
    def possible_dir(self):
        return [1, 2, 3, 4, 6, 7, 8, 9]

    # value of each direction
    def dir_value(self, dir):
        dir_value_table = {
            1: (-1, -1),
            2: (-1, 0),
            3: (-1, 1),
            4: (0, -1),
            6: (0, 1),
            7: (1, -1),
            8: (1, 0),
            9: (1, 1)
        }
        return dir_value_table[dir]

    # probe the direction until reach the boundary of the map or hit something that is not 0
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

        newPlayerID = playerID % 4 + 1

        return (newPlayerID, newMapStat, newSheepStat)

    # return all possible actions
    def get_possible_actions(self, state):
        playerID, mapStat, sheepStat = state
        actions = []
        height, width = len(mapStat), len(mapStat[0])
        # find all splitable sheep group and append the groups' index to splitable_sheep
        splitable_sheep = []
        for i in range(height):
            for j in range(width):
                if mapStat[i][j] == playerID and int(sheepStat[i][j]) > 1:
                    splitable_sheep.append((i, j))
        for i, j in splitable_sheep:
            for dir in self.possible_dir():
                newx, newy = self.probe_direction(i, j, dir, mapStat)
                if newx == i and newy == j:
                    continue
                m = int(sheepStat[i][j]) // 2
                actions.append([(i, j), m, dir])

        return actions

    # check if the state is leaf
    def is_leaf(self, state):
        return not self.get_possible_actions(state)

    # check if the game is end
    def is_end(self, state):
        for i in range(1, 5):
            if self.get_possible_actions((i, state[1], state[2])):
                return False
        return True

    def get_winner(self, state):
        # calculate the territory of each player
        playerID, mapStat, sheepStat = state
        territory = {1: 0, 2: 0, 3: 0, 4: 0}
        for i in range(len(mapStat)):
            for j in range(len(mapStat[0])):
                if 1 <= mapStat[i][j] <= 4:
                    territory[mapStat[i][j]] += 1
        # find the player with the largest territory
        winner = max(territory, key=territory.get)
        return winner

class Node:
    def __init__(self, state, action=None, parent=None):
        self.state = state
        self.action = action  # action applied to get this node from the parent node
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
        self.playerID = state[0]

    def select(self, node):
        while node.children:
            node = max(node.children, key=lambda n: self.ucb(n))
        return node

    def expand(self, node):
        actions = self.get_possible_actions(node.state)
        for action in actions:
            new_state = self.apply_action(node.state, action)
            new_node = Node(new_state, action, node)
            node.children.append(new_node)

    def simulate(self, state):
        # simulate the game until the end
        while not GameInteraction().is_end(state):
            actions = GameInteraction().get_possible_actions(state)
            if not actions:
                state = (state[0] % 4 + 1, state[1], state[2])
                continue
            action = random.choice(actions)
            state = GameInteraction().apply_action(state, action)
        winner = GameInteraction().get_winner(state)
        return 1 if winner == self.playerID else -1

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
        for _ in range(300):
            selected_node = self.select(root)
            if not selected_node.children:
                self.expand(selected_node)

            expanded_node = selected_node if GameInteraction().is_leaf(selected_node.state) else random.choice(selected_node.children)
            value = self.simulate(expanded_node.state)
            self.backpropagate(expanded_node, value)

        best_child = max(root.children, key=lambda n: n.visits) # best child is the node with the most visits
        return best_child.action

    def get_possible_actions(self, state):
        return GameInteraction().get_possible_actions(state)

    def apply_action(self, state, action):
        return GameInteraction().apply_action(state, action)

'''
    選擇起始位置
    選擇範圍僅限場地邊緣(至少一個方向為牆)
    
    return: init_pos
    init_pos=[x,y],代表起始位置
    
'''
def InitPos(mapStat):
    init_pos = [0, 0]
    height, width = len(mapStat), len(mapStat[0])
    
    return init_pos

def GetStep(playerID, mapStat, sheepStat):
    mcts = MCTS((playerID, mapStat, sheepStat))
    action = mcts.get_action()
    return action

# player initial
(id_package, playerID, mapStat) = STcpClient.GetMap()
init_pos = InitPos(mapStat)
STcpClient.SendInitPos(id_package, init_pos)

# start game
while (True):
    (end_program, id_package, mapStat, sheepStat) = STcpClient.GetBoard()
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)

    STcpClient.SendStep(id_package, Step)
