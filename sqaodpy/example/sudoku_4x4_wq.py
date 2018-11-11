# Ref. https://qiita.com/gyu-don/items/0c59423b12256821c38c
# Notebook : https://github.com/gyu-don/wildcat/blob/master/Number%20Place%204x4.ipynb

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

def run(command, number) :
    elapsed_time = timeit.timeit(command, globals = globals(), number=number)
    print('  time[s] = {}'.format(elapsed_time / number))
    print('  E = {}'.format(a.E[-1]))
    print('  solution')
    show_board(a.result)
    print


# solve
import timeit

# wildqat
import wildqat as wq
print('wildcat(sa)')
a = wq.opt()
a.qubo = qubo.tolist()
run('a.result = a.sa()', 1)

#run('a.result = a.sqa()[0]', 1)
#a.plot()

# sqaod
import sqaod as sq
import sqaod.wildqat as swq

a = swq.opt(pkg = sq.cpu)
a.qubo = qubo

print('sqaod_wildcat(sa)')
run('a.result = a.sa()', 10)
print('sqaod_wildcat(sqa)')
run('a.result = a.sqa()[0]', 10)
