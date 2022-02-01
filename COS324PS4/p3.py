
import numpy as np
import matplotlib.pyplot as plt

dimension = 10000
points = 100

mean = np.zeros(dimension)
cov = np.identity(dimension)
data = np.random.multivariate_normal(mean, cov, points)
lsquared = np.sum(np.abs(data)**2, axis=-1)

plt.title("Sample Distribution for d=1000")
plt.xlabel("Squared distance")
plt.ylabel("Probability")
plt.hist(lsquared, bins=500)
plt.show()