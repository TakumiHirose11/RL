import copy

import numpy as np
import mojimoji

class Node:
    def __init__(self, action, reward=0):
        # up, down, right, left
        self.action = action
        self.reward = reward
        self.is_visited = False

class GridWorld:
    def __init__(self, h: int, w: int, start: list, goal: list, reward: np.array, max_tern=1e7):
        self.h = h
        self.w = w
        self.max_tern = max_tern
        self.start = start
        self.goal = goal
        self.reward = reward
        self.board = self.init_board()


    def init_board(self):
        board = [[None] * self.w] * self.h
        for h_i in range(self.h):
            for w_i in range(self.w):
                node_i = Node(action=[1, 1, 1, 1], reward=self.reward[h_i][w_i])
                # up
                if h_i == self.h - 1:
                    node_i.action[0] = 0
                # down
                if h_i == 0:
                    node_i.action[1] = 0
                # right
                if w_i == 0:
                    node_i.action[2] = 0
                # left
                if w_i == self.w - 1:
                    node_i.action[3] = 0
                board[h_i][w_i] = copy.deepcopy(node_i)
                del node_i
        return board

    def show(self):
        max_l = max(len(str(np.max(self.reward))), len(str(np.min(self.reward))))+2
        graph = "ー" + "ー" * (max_l+1) * self.w + "\n"
        for h_i in range(self.h):
            graph += "｜" + ("　" * max_l + "｜")*self.w + "\n"
            graph += "｜"
            for w_i in range(self.w):
                r = mojimoji.han_to_zen(str(self.reward[h_i][w_i]))
                space_l = int((max_l - len(r) + 1)/2)
                graph += "　" * space_l + r + "　"*(max_l - space_l - len(r)) + "｜"
            graph += "\n"
            graph += "｜" + ("　" * max_l + "｜") * self.w + "\n"
            graph += "ー" + "ー" * (max_l + 1) * self.w + "\n"
        print(f"start:{self.start}")
        print(f"goal:{self.goal}")
        print(graph)


if __name__ == '__main__':
    reward = np.array([
            [0, 30, 2, 8],
            [7, 10, 5, -20],
            [-30, -100, 10, 100], ])

    GW = GridWorld(3, 4, [0, 0], [2, 3], reward)
    print(GW.board[0][3].action)
    print(GW.board[2][3].action)




