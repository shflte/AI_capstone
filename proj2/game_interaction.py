import numpy as np

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

    def get_territory(self, mapStat):
        territory = {1: 0, 2: 0, 3: 0, 4: 0}
        for i in range(len(mapStat)):
            for j in range(len(mapStat[0])):
                if 1 <= mapStat[i][j] <= 4:
                    territory[mapStat[i][j]] += 1
        return territory

    def get_winner(self, state):
        playerID, mapStat, sheepStat = state
        territory = self.get_territory(mapStat)
        max_territory = max(territory.values())
        winners = [k for k, v in territory.items() if v == max_territory]
        return winners[0]

    # generate mock sheepStat for game setting 3
    def mock_sheep_stat(self, mapStat):
        territory = self.get_territory(mapStat)
        sheepDict = {}
        for i in range(1, 5):
            length = territory[i]
            base = 16 // length
            remainder = 16 % length
            sheepDict[i] = [base + 1 if i < remainder else base for i in range(length)]

        mockSheepStat = np.zeros((12, 12))
        # assign sheep to each player's territory
        for i in range(12):
            for j in range(12):
                if 1 <= mapStat[i][j] <= 4:
                    playerID = mapStat[i][j]
                    mockSheepStat[i][j] = sheepDict[playerID].pop()

        return mockSheepStat
