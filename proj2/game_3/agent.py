import STcpClient
import numpy as np

import sys
sys.path.append('..')
from MCTS import MCTS
from game_interaction import GameInteraction

def InitPos(mapStat):
    init_pos = [0, 0]
    height, width = len(mapStat), len(mapStat[0])

    return init_pos

def GetStep(playerID, mapStat, sheepStat):
    # sheep stat is hidden, call mock sheep stat
    mockSheepStat = GameInteraction().mock_sheep_stat((playerID, mapStat, sheepStat))
    mcts = MCTS((playerID, mapStat, mockSheepStat), 3)
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
    sheepStat = np.where(mapStat == playerID, sheepStat, 0) # hide other player's sheep number
    # hide other player's sheep number
    if end_program:
        STcpClient._StopConnect()
        break
    Step = GetStep(playerID, mapStat, sheepStat)

    STcpClient.SendStep(id_package, Step)
# DON'T MODIFY ANYTHING IN THIS WHILE LOOP OR YOU WILL GET 0 POINT IN THIS QUESTION