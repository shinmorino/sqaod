# Ref. https://qiita.com/gyu-don/items/0c59423b12256821c38c
# Notebook : https://github.com/gyu-don/wildcat/blob/master/Number%20Place%204x4.ipynb

from __future__ import print_function

problem_str = '''
1? ??
?? 2?

?3 ??
?? ?4'''

# Indices of array are:
#  0  1 |  2  3
#  4  5 |  6  7
# ------+-------
#  8  9 | 10 11
# 12 13 | 14 15

# Remove whitespace and newline
import re
problem_str = re.sub(r'[^0-9?]', '', problem_str)

problem_str

import numpy as np

n = 4**3
grouping = [
    # rows
    [ 0,  1,  2,  3],
    [ 4,  5,  6,  7],
    [ 8,  9, 10, 11],
    [12, 13, 14, 15],
    # columns
    [ 0,  4,  8, 12],
    [ 1,  5,  9, 13],
    [ 2,  6, 10, 14],
    [ 3,  7, 11, 15],
    # blocks
    [ 0,  1,  4,  5],
    [ 2,  3,  6,  7],
    [ 8,  9, 12, 13],
    [10, 11, 14, 15],
]


# Bits arrangement:
# i = 0..15: i-th value is 1
# i = 16..31: (i-16)-th value is 2
# i = 32..47: (i-32)-th value is 3
# i = 48..63: (i-48)-th value is 4

# 1. Symmetrize
# 2. Beta, kT
# 3. dE calc

qubo = np.eye(n) * -1

for g in grouping:
    for i in range(4):
        for j in range(3):
            for k in range(j+1, 4):
                qubo[i*16 + g[j], i*16 + g[k]] += 2

for i in range(16):
    for j in range(3):
        for k in range(j+1, k):
            qubo[i+j*16, i+k*16] += 2.5

for i, v in enumerate(problem_str):
    if v == '?':
        continue
    v = int(v) - 1
    qubo[v*16 + i, v*16 + i] -= 20
    for j in range(4):
        qubo[j*16 + i, j*16 + i] += 4
    for g in grouping:
        if i not in g:
            continue
        for j in g:
            qubo[v*16 + j, v*16 + j] += 3 + v*0.3

def show_board(result) :
    board = ['?'] * 16
    for i,b in enumerate(result):
        if b == 1:
            if board[i%16] != '?':
                print('!', end='')
            board[i%16] = (i//16) + 1

    print('''  {}{} {}{}
  {}{} {}{}
  {}{} {}{}
  {}{} {}{}'''.format(*board))

import sqaod as sq

ann = sq.cpu.dense_graph_annealer(qubo, dtype = np.float32)

print('sqaod_wildcat(sa)')

T, Tfin = 5, 0.02
_ = None # '_'  is ignored in get_system_E() and anneal_one_step().
tau = 0.998

ann.set_preferences(n_trotters = 1, algorithm = sq.algorithm.sa_default)
ann.prepare()
ann.randomize_spin()

Esa_qubo = [ann.get_system_E(_, _)]
while T > Tfin :
    ann.anneal_one_step(T, _)
    T *= tau
    Esa_qubo.append(ann.get_system_E(_, _))
    
x = ann.get_x()
print('E(QUBO) = ', Esa_qubo[-1])
show_board(x[0])


print('sqaod_wildcat(sqa)')

G, Gfin = 5, 0.02
beta = 1. / 0.05
tau = 0.96

ann.set_preferences(n_trotters = 8, algorithm = sq.algorithm.default)
ann.prepare()
ann.randomize_spin()

Esqa = [ann.get_system_E(G, beta)]
Esqa_qubo = [np.mean(ann.get_E())]
while G > Gfin :
    ann.anneal_one_step(G, beta)
    G *= tau
    Esqa.append(ann.get_system_E(G, beta))
    Esqa_qubo.append(np.mean(ann.get_E()))

x = ann.get_x()
print('E(SQA)  = ', Esqa[-1])
print('E(QUBO) = ', Esqa_qubo[-1])

show_board(x[0])

import matplotlib.pyplot as plt
plt.figure(figsize=(8,10))
plt.subplot(211)
plt.title('Simulated annealing')
plt.ylabel('E')
h_qubo, = plt.plot(Esa_qubo, label='E(QUBO)')
plt.legend(handles=[h_qubo])

plt.subplot(212)
plt.title('Simulated quantum annealing')
plt.ylabel('E')
h_sqa, = plt.plot(Esqa, label = 'E(System)')
h_qubo, = plt.plot(Esqa_qubo, label = 'E(QUBO)')
plt.legend(handles=[h_sqa, h_qubo])
plt.show()
