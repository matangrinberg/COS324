import numpy as np
import matplotlib.pyplot as plt

grid = np.array([[0, 0, 0, 0, 0, 0],
                 [0, 7, 12, 1, 14, 0],
                 [0, 2, 13, 8, 11, 0],
                 [0, 16, 3, 10, 5, 0],
                 [0, 9, 6, 15, 4, 0],
                 [0, 0, 0, 0, 0, 0]])

aims = np.array([[1, 1], [1, 2], [1, 3], [1, 4],
                 [2, 1], [2, 2], [2, 3], [2, 4],
                 [3, 1], [3, 2], [3, 3], [3, 4],
                 [4, 1], [4, 2], [4, 3], [4, 4]])


def shot_outcome(aim):
    row = aim[0]
    column = aim[1]
    possibilities = [grid[row, column], grid[row + 1, column],
                     grid[row - 1, column], grid[row, column + 1], grid[row, column - 1]]
    result = np.random.choice(possibilities, p=[0.6, 0.1, 0.1, 0.1, 0.1])
    return result


def reward(score):
    return -np.heaviside(score - 101, -1)


# T is how many steps until completion

def value_iteration(T):
    v = np.zeros([T+1, 117])
    v[:, 101] = np.ones(T+1)
    v[:, 102:117] = -np.ones([T+1, 15])
    p = np.zeros([T+1, 117])

    for k in range(1, T+1):
        for s in range(101):
            q = np.zeros([16])
            for a in range(len(aims)):
                hit = s + grid[aims[a, 0], aims[a, 1]]
                north = s + grid[aims[a, 0] + 1, aims[a, 1]]
                east = s + grid[aims[a, 0], aims[a, 1] + 1]
                west = s + grid[aims[a, 0], aims[a, 1] - 1]
                south = s + grid[aims[a, 0] - 1, aims[a, 1]]
                q[a] = 0.6*v[k-1, hit] + 0.1*(v[k-1, north] + v[k-1, east] + v[k-1, west] + v[k-1, south])

            p[k, s] = np.argmax(q)
            v[k, s] = np.max(q)
    return v

val = value_iteration(100)
data = val[100, 0:102]

plt.plot(data)
plt.title("Value Function")
plt.xlabel("Score")
plt.ylabel("Value")
plt.show()
