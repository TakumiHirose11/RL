import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML

"""迷路関連"""
# 次状態の取得
def get_s_next(s, a):
    if possible_a[s, a] == 0:
        return None
    if a==0:
        return s-3
    elif a==1:
        return s+1
    elif a==2:
        return s+3
    elif a==3:
        return s-1
    
#可能な行動のリスト
possible_a = np.array([  #上、右、下、左
    [0, 1, 1, 0],      #0
    [0, 1, 1, 1],           #1
    [0, 0, 0, 1], #2
    [1, 0, 1, 0],      #3
    [1, 1, 0, 0],      #4
    [0, 0, 1, 1],      #5
    [1, 1, 0, 0],      #6
    [0, 0, 0, 1], #7
    [1, 1, 0, 0],      #8
])

"""方策関連"""
#ソフトマックス方策
def soft_max_policy(theta:np.array, beta=1):
    [s, a] = theta.shape #s, a : 状態、行動の数
    pi = np.zeros((s,a)) 
    exp_theta = np.exp(theta * beta)
    for i in range(0, s):
        pi[i, :] = exp_theta[i, :] / np.nansum(exp_theta[i, :])
    pi = np.nan_to_num(pi)
    return pi

#target policy
def pi_target(pi):
    pi_d = np.zeros(pi.shape[0])
    for s in range(pi.shape[0]):
        pi_s = pi[s, :]
        pi_d[s] = np.argmax(pi_s)
    return pi_d


def epsilon_greedy(Q, eps=0.3):
    pi = np.zeros_like(Q)
    for s in range(Q.shape[0]):
        Q_s = [Q[s,j] if possible_a[s,j]==1 else 0 for j in range(Q.shape[1])]
        #行動が一つしかとれない場合の例外処理
        if np.count_nonzero(possible_a[s,:] == 1) == 1:
            pi[s, np.argmax(possible_a[s,:])] = 1
            continue
        #一般的な処理
        for a in range(Q.shape[1]):
            if possible_a[s,a] == 0:
                continue
            else:
                pi[s, a] = eps / np.count_nonzero(possible_a[s,:] == 1)
        pi[s, np.argmax(Q_s)] += 1 - eps  
    return pi

"""グラフ関連"""
def make_subplot(graph, place, v, label_list=None, title=None):
    """
    graph = plt.figure()
    place = [1,2,1]
    """
    g = graph.add_subplot(place[0], place[1], place[2])
    if title != None:
            g.set_title(title)
    for i, vi in enumerate(v):
        x = np.arange(len(vi))
        g.plot(x, vi, label=label_list[i])
    g.legend()