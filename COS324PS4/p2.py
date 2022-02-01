
import numpy as np
import math as m
import scipy
import matplotlib.pyplot as plt


def ratio(d):
    temp1 = np.power(np.pi, d/2)
    temp2 = np.power(2, d) * m.gamma(d - 1/2)
    return temp1/temp2


ratioV = np.vectorize(ratio)

xRange = np.linspace(.1, 100.0, 1500)
yRange = ratioV(xRange)

logX = np.log(xRange)
logY = np.log(yRange)

plt.plot(logX, logY)
plt.xlabel('Log(d)')
plt.ylabel('Log(R)')
plt.title('hypersphere-hybercube volume ratio, R, in spatial dimension, d')
plt.show()
