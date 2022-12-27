import numpy as np
import mojimoji

class Node:
    def __init__(self, up, down, left, right, reward=0):
        self.up = up
        self.down = down
        self.left = left
        self.right = right
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
        self.board = self.init_board(h, w, reward)


    def init_board(self, h:int, w:int, reward:np.array):
        board = [[None] * w] * h
        for h_i in range(h):
            for w_i in range(w):
                node_i = Node(1, 1, 1, 1, reward[h_i][w_i])
                if h_i == 0:
                    node_i.up = 0
                if h_i == h - 1:
                    node_i.down = 0
                if w_i == 0:
                    node_i.left = 0
                if w_i == w - 1:
                    node_i.right = 0
                board[h_i][w_i] = node_i
        return board

    def show(self):
        max_l = max(len(str(np.max(self.reward))), len(str(np.min(self.reward))))+2
        graph = "ー" + "ー" * (max_l+1) * self.w + "\n"
        for h_i in range(self.h):
            graph += "｜" + ("　" * max_l + "｜")*self.w + "\n"
            graph += "｜"
            for w_i in range(self.w):
                r = mojimoji.han_to_zen(str(reward[h_i][w_i]))
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
    GW.show()



