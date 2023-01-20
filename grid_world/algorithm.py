from board import GridWorld
import numpy as np
import mojimoji
import copy

class ValueIterationMethod:
    def __init__(self, gw, gamma, ipsilon):
        self.gw = gw
        self.gamma = gamma
        self.ipsilon = ipsilon
        # value function
        self.v_past = np.zeros_like(gw.board)
        self.v_now = np.zeros_like(gw.board)
        self.policy = None

    def update(self):
        self.v_past = copy.deepcopy(self.v_now)
        d = [[1, 0], [-1, 0], [0, -1], [0, 1]]
        for h_i in range(self.gw.h):
            for w_i in range(self.gw.w):
                action = self.gw.board[h_i][w_i].action
                v_list = []
                for i, a in enumerate(action):
                    if a == 1:
                        v_list.append(gw.board[h_i+d[i][0]][w_i+d[i][1]].reward
                                      + self.gamma * self.v_now[h_i+d[i][0]][w_i+d[i][1]])
                v_list = np.array(v_list)
                self.v_now[h_i][w_i] = np.max(v_list)

    def is_end(self):
        residuals = self.v_now - self.v_past
        return np.max(residuals) < self.ipsilon

    def get_policy(self):
        policy = np.zeros_like(gw.board)
        d = [[1, 0], [-1, 0], [0, -1], [0, 1]]
        for h_i in range(self.gw.h):
            for w_i in range(self.gw.w):
                action = self.gw.board[h_i][w_i].action
                v_list = [0] * 4
                for i, a in enumerate(action):
                    if a == 1:
                        v_list[i] = gw.board[h_i + d[i][0]][w_i + d[i][1]].reward \
                                    + self.gamma * self.v_now[h_i + d[i][0]][w_i + d[i][1]]
                v_list = np.array(v_list)
                policy[h_i][w_i] = np.argmax(v_list)
        self.policy = policy
        return policy

    def show(self,mode="v"):
        max_l = None
        if mode == "v":
            max_l = max(len(str(int(np.max(self.v_now)))), len(str(int(np.min(self.v_now)))))+2
        elif mode == "p":
            max_l = max(len(str(int(np.max(self.policy)))), len(str(int(np.min(self.policy)))))+2
        graph = "ー" + "ー" * (max_l+1) * self.gw.w + "\n"
        for h_i in range(self.gw.h):
            graph += "｜" + ("　" * max_l + "｜")*self.gw.w + "\n"
            graph += "｜"
            for w_i in range(self.gw.w):
                r = None
                if mode == "v":
                    r = mojimoji.han_to_zen(str(int(self.v_now[h_i][w_i])))
                elif mode == "p":
                    r = mojimoji.han_to_zen(str(self.policy[h_i][w_i]))
                space_l = int((max_l - len(r) + 1)/2)
                graph += "　" * space_l + r + "　"*(max_l - space_l - len(r)) + "｜"
            graph += "\n"
            graph += "｜" + ("　" * max_l + "｜") * self.gw.w + "\n"
            graph += "ー" + "ー" * (max_l + 1) * self.gw.w + "\n"
        # print(f"start:{self.gw.start}")
        # print(f"goal:{self.gw.goal}")
        print(graph)

    def learning(self):
        while(True):
            self.update()
            if self.is_end():
                break
        return self.get_policy()


if __name__ == "__main__":
    reward = np.array([
        [0, 30, 2, 8],
        [7, 10, 5, -20],
        [-30, -100, 10, 100], ])

    gw = GridWorld(3, 4, [0, 0], [2, 3], reward)
    vim = ValueIterationMethod(gw, gamma=0.8, ipsilon=20)
    vim.learning()
    gw.show()
    vim.show(mode="v")
    vim.show(mode="p")
    print(gw.board[2][3].action)
    print(gw.board[0][3].action)


        