import numpy as np
import random
from game_interaction import GameInteraction

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
        for _ in range(300):
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
