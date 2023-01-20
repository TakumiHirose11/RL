# %% [markdown]
# # Dynamic Planning  
# 分類 : 価値反復法
# -  モデルベース  
# -  価値ベース  

# %%
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from matplotlib import animation
from IPython.display import HTML

# %%
#迷路作成
fig = plt.figure(figsize=(3,3))

#壁
plt.plot([0, 3], [3, 3], color="k")
plt.plot([0, 3], [0, 0], color="k")
plt.plot([0, 0], [0, 2], color="k")
plt.plot([3, 3], [1, 3], color="k")
plt.plot([1, 1], [1, 2], color="k")
plt.plot([2, 3], [2, 2], color="k")
plt.plot([2, 1], [1, 1], color="k")
plt.plot([2, 2], [0, 1], color="k")

#数字
for i in range(3):
    for j in range(3):
        plt.text(0.5+i, 2.5-j, str(i+j*3), size=20, ha="center", va="center")

#円
circle, = plt.plot([0.5], [2.5], marker="o", color="#d3d3d3", markersize="40")

#メモリと枠の非表示
plt.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="off", right="off", left="off", labelleft="off")
plt.box("off")
plt.gca().axes.xaxis.set_visible(False)
plt.gca().axes.yaxis.set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)



# %% [markdown]
# ## 環境についての情報

# %%
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

# p_t = 1

g = np.zeros((9,4))
g[5,2] = 1 

# %%
def get_s_next(s, a):
    if a==0:
        return s-3
    elif a==1:
        return s+1
    elif a==2:
        return s+3
    elif a==3:
        return s-1

# %%
def B_pi(Q, pi, ganma=0.9):
    pi_table = pi(Q=Q)
    Q_next = Q.copy()
    for s in range(Q.shape[0]):
        for a in range(Q.shape[1]):
            Q_next[s,a] = g[s,a] + np.sum(pi_table[get_s_next(s,a),:].T * Q[get_s_next(s,a), :])
    delta = np.sum(Q_next - Q)
    return Q_next, delta
    

# %%
def B_asterisk(Q, ganma=0.9):
    Q_next = Q.copy()
    for s in range(Q.shape[0]):
        for a in range(Q.shape[1]):
            Q_next[s,a] = g[s,a] + ganma * np.argmax(Q[get_s_next(s,a), :])
    delta = np.sum(Q_next - Q)
    return Q_next, delta

# %%
def learning_B_asterisk(ganma=0.9, epoch=1000, stop_epsilon=10e-3, pi=None):
    Q = np.zeros((9,4))
    delta_l = []
    for episode in range(epoch):
        Q, delta = B_asterisk(Q=Q, ganma=ganma)
        delta_l.append(delta)
        if delta < stop_epsilon:
            break
    return Q, delta_l

def learning_B_pi(ganma=0.9, epoch=1000, stop_epsilon=10e-3, pi=None):
    Q = np.zeros((9,4))
    delta_l = []
    for episode in range(epoch):
        Q, delta = B_pi(Q=Q, ganma=ganma, pi=pi)
        delta_l.append(delta)
        if delta < stop_epsilon:
            break
    return Q, delta_l

# %%
#最終的な方策
def pi_d(pi):
    pi_d = np.zeros(pi.shape[0])
    for s in range(pi.shape[0]):
        pi_s = pi[s, :]
        pi_d[s] = np.argmax(pi_s)
    return pi_d

# %%
Q_pi, pi_delta = learning_B_pi(B_pi, pi=epsilon_greedy)
Q_asterisk, as_delta = learning_B_asterisk(B_asterisk)

# %%
pi_d1 = pi_d(Q_pi)
pi_d2 = pi_d(Q_asterisk)

print(Q_pi)
print(Q_asterisk)

print(pi_d1)
print(pi_d2)

# %% [markdown]
# - 過学習どころか、学習しすぎると悪化する（勾配の定義からループに入ったらやばそうなきがしていたが...）
# - 価値に訪問回数を用いると、ループの価値が増えてしまう
# - 逆温度の値でランダム性を高めても改善は見られない。グラフの形は結構変わるが。
# - myの方が早く収束するがループにハマる回数が多い、本の方がわずかにまし。



# %%
""" def animate(i):
    state = s_a_history[i][0]
    circle.set_data((state % 3) + 0.5, 2.5 - int(state/3))
    return circle

anim = animation.FuncAnimation(fig, animate, frames=len(s_a_history), interval=200, repeat=False)
HTML(anim.to_jshtml()) """


