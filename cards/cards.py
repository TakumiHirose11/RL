import numpy as np
class Card:
    def __init__(self, mark: int, number: int):
        # Spade:0, Club:1, Diamond:2, Heart:3
        self.mark = mark
        # 1-13
        self.number = number

class SevenRows:
    def __init__(self, n_player=4, joker=0, log=False, pass_count=3):
        # number of player
        self.n_player = n_player
        # cards on table   
        self.table = np.array([[0]*13]*4)
        # cards of each player, player_i, mark_i, num_i
        self.hands = np.array([[[0]*13]*4]*n_player)
        # now turn
        self.player = None
        # count you can pass
        self.pass_count = pass_count
        self.log = log
        self.joker = joker

    def deal_cards(self):
        num_of_cards = [int(52/self.n_player)] * self.n_player
        for i in range(52 - int(52/self.n_player) * self.n_player):
            num_of_cards[i] += 1
        cards = np.array([(i, j) for i in range(4) for j in range(13)])
        np.random.shuffle(cards)
        for player_i in range(self.n_player):
            sum = 0
            for i in range(sum, sum + num_of_cards[player_i]):
                self.hands[player_i][cards[i][0]][cards[i][1]] = 1
        del sum, cards, num_of_cards
        if self.log:
            print("Cards dealt.")

    def init_process(self):
        # 7を持っている人の処理と、ダイヤの７が最初のプレイヤー
        for player_i in range(self.n_player):
            for i in range(4):
                if self.hands[player_i][i][6]:
                    self.hands[player_i][i][6] = 0
                    self.table[i][6] = 1
                    if i == 2:
                        self.player = i
        if self.log:
            print("Initialized.")
            print(f"First player is {self.player}.")
            print("Enter action (mark, number).")

    def next_state(self, mark, num):
        if not self.hands[self.player][mark][num]:
            print("you don't have this card.")
            return
        self.hands[self.player][mark][num] = 0
        self.table[mark][num] = 1
        self.player = (self.player + 1) % self.n_player





sr = SevenRows()
sr.deal_cards()
sr.init_process()



