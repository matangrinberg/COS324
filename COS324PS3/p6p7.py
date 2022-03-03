
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt

rawData = pn.read_csv('motorcycle.csv')
data = np.transpose(rawData.as_matrix())

times = data[0]/60
forces = data[1]

theta = np.random.normal(0, 1, [10, 15])
weights = np.random.normal(0, 1, [10, 15])
biases = np.random.normal(0, 1, [10, 15])
xRange = np.transpose((1/100)*(np.arange(0, 1000)-500))


def nl1(x, th, b):
    vec = np.multiply(x, th) + b
    temp1 = np.tanh(vec)
    return np.inner(temp1, th)


def nl2(x, th, b):
    vec = np.multiply(x, th) + b
    temp2 = np.maximum(vec, 0)
    return np.inner(temp2, th)


def nl3(x, th, b):
    vec = np.multiply(x, th) + b
    temp3 = np.exp(-np.power(vec, 2))
    return np.inner(temp3, th)


def nlfunc1(vec):
    return np.tanh(vec, 2)


def nlfunc2(vec):
    return np.maximum(0,vec)


def nlfunc3(vec):
    return np.exp(-np.power(vec, 2))


def phi(func, x, th, b):
    v_func = np.vectorize(func)
    return v_func(np.multiply(x, th) + b)


def avg_gradient_w(func, w, th, b, xVec, yVec):
    s = np.zeros(len(w))
    for i in range(0, len(xVec)):
        ydiff = np.inner(w, phi(func, xVec[i], th, b)) - yVec[i]
        s += np.multiply(ydiff, phi(func, xVec[i], th, b))

    return s


def avg_gradient_b(func, w, th, b, xVec, yVec):
    s = np.zeros(len(w))
    for i in range(0, len(xVec)):
        s += np.inner(w, phi(func, xVec[i], th, b)) - yVec[i]

    return s

# exploits that fact that Jacobian is diagonal


def jacob1(th, b, x):
    va = x
    j = []
    for i in range(0, len(b)):
        vb = th[i]
        vc = b[i]
        v1 = va * vb
        v2 = vc
        v3 = v1+v2
        v4 = -np.power(v3, 2)
        v5 = np.exp(v4)
        v5b = 1
        v4b = v5b * v5
        v3b = v4b * (-2 * v3)
        v1b = v3b
        vbb = v1b * va
        theta_bar = vbb
        j.append(theta_bar)

    return j


def avg_gradient_th(func, w, th, b, xVec, yVec):
    s = np.zeros(len(w))
    for i in range(0, len(xVec)):
        ydiff = np.inner(w, phi(func, xVec[i], th, b)) - yVec[i]
        j = jacob1(func, th, b, xVec[i])
        s += np.multiply(j, np.multiply(ydiff, w))

    return s


def loss(real, prediction):
    return np.inner(real - prediction, real - prediction)


bstar = np.random.normal(0, 1, 15)
wstar = np.random.normal(0, 1, 15)
thstar = np.random.normal(0, 1, 15)
alpha = 0.0001

for i in range(0, 10):
    bstar -= np.multiply(alpha, avg_gradient_b(nlfunc3, wstar, thstar, bstar, times, forces))
    wstar -= np.multiply(alpha, avg_gradient_w(nlfunc3, wstar, thstar, bstar, times, forces))
    thstar -= np.multiply(alpha, avg_gradient_b(nlfunc3, wstar, thstar, bstar, times, forces))
    print(bstar)


def pred(b, w, th, x):
    return np.inner(w, phi(nlfunc3, x, th, b))


nl1vec = np.vectorize(nl1)
nl2vec = np.vectorize(nl2)
nl3vec = np.vectorize(nl3)


for i in range(0, len(times)):
    #plt.scatter(times[i], pred(bstar, wstar, thstar, times[i]), marker = 'o')
    plt.scatter(times[i], forces[i], marker='o')

# for i in range(0, 10):
#     plt.plot(xRange, nl1vec(xRange, i))
#
# plt.title('Neural network, J=15, hyperbolic tangent nonlinearity')

# for i in range(0, 10):
#     plt.plot(xRange, nl2vec(xRange, i))
#
# plt.title('Neural network, J=15, rectified linear units')

# for i in range(0, 10):
#     plt.plot(xRange, nl3vec(xRange, i))
#
# plt.title('Neural network, 15 hidden units, RBF nonlinearity')

plt.show()
