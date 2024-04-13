'''
    Team 23 : 前物連網無念容
    Members : 109550100 陳宇駿 109550085 陳欣妤 109550106 涂圓緣
'''
import STcpClient

import numpy as np
import random

class GameInteraction:
    def __init__(self, board_size=12):
        self.board_size = board_size

    # flip row and column of the game state (mapStat, sheepStat)
    def flip_board(self, state):
        playerID, mapStat, sheepStat = state
        newMapStat = mapStat.copy()
        newSheepStat = sheepStat.copy()
        for i in range(len(mapStat)):
            for j in range(len(mapStat[0])):
                newMapStat[j][i] = mapStat[i][j]
                newSheepStat[j][i] = sheepStat[i][j]
        return (playerID, newMapStat, newSheepStat)

    def flip_pos(self, pos):
        x, y = pos
        return (y, x)

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

    def check_valid_init(self, mapStat, init_pos): 
        x, y = init_pos

        if mapStat[x][y] != 0:
            return False

        extended_map = np.pad(mapStat.copy(), pad_width=1, mode='constant', constant_values=0)
        window = extended_map[x:x+3, y:y+3]
        return np.any(window == -1)

    def on_board(self, x, y, board_size):
        return (x >= 0) and (x < board_size) and (y >= 0) and (y < board_size)

    def get_init_pos(self, mapStat, board_size=12):
        surroundings = [
            (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2),
            (-1, -2), (-1, -1), (-1, 0), (-1, 1), (-1, 2),
            (0, -2),  (0, -1),           (0, 1),  (0, 2),
            (1, -2),  (1, -1),  (1, 0),  (1, 1),  (1, 2),
            (2, -2),  (2, -1),  (2, 0),  (2, 1),  (2, 2)
        ]

        max_count = -1
        best_init_pos = [0, 0]

        for i in range(board_size):
            for j in range(board_size):
                init_pos = [i, j]

                if not self.check_valid_init(mapStat, init_pos):
                    continue

                count = 0
                for dx, dy in surroundings:
                    nx, ny = init_pos[0] + dx, init_pos[1] + dy
                    if self.on_board(nx, ny, board_size) and mapStat[nx][ny] == 0:
                        count += 1

                if count > max_count:
                    max_count = count
                    best_init_pos = init_pos

        return best_init_pos

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

    def get_territory(self, mapStat):
        territory = {1: 0, 2: 0, 3: 0, 4: 0}
        for i in range(len(mapStat)):
            for j in range(len(mapStat[0])):
                if 1 <= mapStat[i][j] <= 4:
                    territory[mapStat[i][j]] += 1
        return territory

    def dfs(self, mapStat, playerID, visited, i, j):
        if i < 0 or i >= self.board_size or j < 0 or j >= self.board_size or visited[i][j] or mapStat[i][j] != playerID:
            return 0
        visited[i][j] = True
        return 1 + self.dfs(mapStat, playerID, visited, i - 1, j) \
            + self.dfs(mapStat, playerID, visited, i + 1, j) \
            + self.dfs(mapStat, playerID, visited, i, j - 1) \
            + self.dfs(mapStat, playerID, visited, i, j + 1)

    def get_connected_regions(self, mapStat, playerID):
        connected_regions = []
        visited = [[False for _ in range(self.board_size)] for _ in range(self.board_size)]
        for i in range(self.board_size):
            for j in range(self.board_size):
                if mapStat[i][j] == playerID and not visited[i][j]:
                    connected_regions.append(self.dfs(mapStat, playerID, visited, i, j))
        return connected_regions

    def get_player_score(self, state):
        playerID, mapStat, sheepStat = state
        regions = self.get_connected_regions(mapStat, playerID)
        return round(sum([region ** 1.25 for region in regions]))

    def get_winner(self, state):
        playerID, mapStat, sheepStat = state
        scores = [self.get_player_score((i, mapStat, sheepStat)) for i in range(1, 5)]
        return scores.index(max(scores)) + 1

    def get_winning_team(self, state):
        scores = [self.get_player_score((i, state[1], state[2])) for i in range(1, 5)]
        return 1 if scores[0] + scores[2] > scores[1] + scores[3] else 2

    # generate mock sheepStat for game setting 3
    def mock_sheep_stat(self, state):
        playerID, mapStat, sheepStat = state
        territory = self.get_territory(mapStat)
        sheepDict = {}
        for i in range(1, 5):
            length = territory[i]
            base = 16 // length
            remainder = 16 % length
            sheepDict[i] = [base + 1 if i < remainder else base for i in range(length)]

        mockSheepStat = sheepStat.copy()
        # assign sheep to each player's territory
        for i in range(self.board_size):
            for j in range(self.board_size):
                if 1 <= mapStat[i][j] <= 4 and mapStat[i][j] != playerID:
                    mockSheepStat[i][j] = sheepDict[mapStat[i][j]].pop()

        return mockSheepStat

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
    def __init__(self, state, game_setting_id):
        self.tree = Tree(state)
        self.playerID = state[0]
        self.game_setting_id = game_setting_id
        self.iterations = 100 if game_setting_id == 2 else 300
        self.game_interaction = GameInteraction(15 if self.game_setting_id == 2 else 12)

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
        while not self.game_interaction.is_end(state):
            actions = self.game_interaction.get_possible_actions(state)
            if not actions:
                state = (state[0] % 4 + 1, state[1], state[2])
                continue
            action = random.choice(actions)
            state = self.game_interaction.apply_action(state, action)

        if self.game_setting_id == 4:
            player_team = 1 if self.playerID in [1, 3] else 2
            winning_team = self.game_interaction.get_winning_team(state)
            return 1 if player_team == winning_team else -1
        else:
            winner = self.game_interaction.get_winner(state)
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
        for _ in range(self.iterations):
            selected_node = self.select(root)
            if not selected_node.children:
                self.expand(selected_node)

            expanded_node = selected_node if self.game_interaction.is_leaf(selected_node.state) else random.choice(selected_node.children)
            value = self.simulate(expanded_node.state)
            self.backpropagate(expanded_node, value)

        best_child = max(root.children, key=lambda n: n.visits) # best child is the node with the most visits
        return best_child.action

    def get_possible_actions(self, state):
        return self.game_interaction.get_possible_actions(state)

    def apply_action(self, state, action):
        return self.game_interaction.apply_action(state, action)


def InitPos(mapStat):
    init_pos = GameInteraction().get_init_pos(mapStat)
    #init_pos = GameInteraction().flip_pos(init_pos)
    return init_pos

def GetStep(playerID, mapStat, sheepStat):
    mcts = MCTS((playerID, mapStat, sheepStat), 4)
    action = mcts.get_action()
    action = GameInteraction().flip_action(action)
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
