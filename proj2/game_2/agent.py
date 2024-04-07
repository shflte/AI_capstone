import STcpClient

import sys
sys.path.append('..')
from MCTS import MCTS
from game_interaction import GameInteraction

def InitPos(mapStat):
    init_pos = GameInteraction(15).get_init_pos(mapStat, 15)
    init_pos = GameInteraction(15).flip_pos(init_pos)
    return init_pos

def GetStep(playerID, mapStat, sheepStat):
    mcts = MCTS((playerID, mapStat, sheepStat), 2)
    action = mcts.get_action()
    action = GameInteraction(15).flip_action(action)
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
