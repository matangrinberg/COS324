
import numpy as np
import pandas as pn
import matplotlib.pyplot as plt

# Data pre-processing

rawTrainData = pn.read_csv("hmeq-train.csv")
hotTrainReason = pn.get_dummies(rawTrainData['REASON'])
hotTrainJob = pn.get_dummies(rawTrainData['JOB'])
rawTestData = pn.read_csv("hmeq-test.csv")
hotTestReason = pn.get_dummies(rawTestData['REASON'])
hotTestJob = pn.get_dummies(rawTestData['JOB'])

for j in range(0, 7):
    rawTrainData.insert(5 + j, hotTrainJob.columns[j], hotTrainJob[hotTrainJob.columns[j]])
    rawTestData.insert(5 + j, hotTestJob.columns[j], hotTestJob[hotTestJob.columns[j]])

for i in range(0, 2):
    rawTrainData.insert(4 + i, hotTrainReason.columns[i], hotTrainReason[hotTrainReason.columns[i]])
    rawTestData.insert(4 + i, hotTestReason.columns[i], hotTestReason[hotTestReason.columns[i]])

rawTrainData.insert(6, 'unknown', hotTrainReason[hotTrainReason.columns[2]])
rawTestData.insert(6, 'unknown', hotTestReason[hotTestReason.columns[2]])
trainData = rawTrainData.drop(columns=['REASON', 'JOB'])
testData = rawTestData.drop(columns=['REASON', 'JOB'])


continuousdata = pn.Series([0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

trainMeans = trainData.mean(axis=0)
trainSTDs = trainData.std(axis=0)
testMeans = testData.mean(axis=0)
testSTDs = testData.std(axis=0)

for j in range(0, 20):
    trainMeans[j] = trainMeans[j] * continuousdata[j]
    trainSTDs[j] = np.power(trainSTDs[j], continuousdata[j])
    testMeans[j] = testMeans[j] * continuousdata[j]
    testSTDs[j] = np.power(testSTDs[j], continuousdata[j])

normTrainData = (trainData - trainMeans)/trainSTDs
normTestData = (testData - testMeans)/testSTDs

Xp = normTrainData.drop(columns=['BAD']).to_numpy()
X = np.hstack((Xp, np.ones((4000, 1))))
y = normTrainData['BAD'].to_numpy()
testXp = normTestData.drop(columns=['BAD']).to_numpy()
testX = np.hstack((testXp, np.ones((1357, 1))))
testY = normTestData['BAD'].to_numpy()


# Likelihood Funciton


def sigma(x):
    return 1/(1+np.exp(-x))


def loglikelihood(xarray, yvec, weights):
    s = sigma(np.dot(xarray, weights))
    return np.dot(yvec, np.log(s)) + np.dot(1-yvec, np.log(1-s))


def avggradient(xarray, yvec, weights):
    s = sigma(np.dot(xarray, weights))
    return np.dot(xarray.transpose(), (yvec - s))/(yvec.shape[0])


def gradascent(xarray, yvec, weights, alpha, n, lam):
    wp = np.ones(weights.size)
    for it in range(0, n):
        wp += alpha * (avggradient(xarray, yvec, wp) - np.divide(lam, float(yvec.shape[0])) * wp)
    return wp


def plotgradascent(xarray, yvec, weights, alpha, n, lam):
    wp = np.ones(weights.size)
    for it in range(0, n):
        wp += alpha * (avggradient(xarray, yvec, wp) - np.divide(lam, float(yvec.shape[0])) * wp)
        plt.plot(it, loglikelihood(xarray, yvec, wp), '.', color='orange')
    plt.title('Log Likelihood, $\\alpha = $' + str(alpha))
    plt.ylabel('Log Likelihood')
    plt.xlabel('Iterations')
    plt.show()


def accuracy(x, dset, weights):
    s = sigma(np.dot(x, weights))
    guesses = np.round(s)
    return np.sum(np.abs(guesses + dset - 1))/dset.size


def plotacc(weights, alpha, n, lam):
    wp = np.ones(weights.size)
    trainAcc = []
    testAcc = []
    for it in range(0, n):
        wp += alpha * (avggradient(X, y, wp) - np.divide(lam, float(y.shape[0])) * wp)
        trainAcc.append(accuracy(X, y, wp))
        testAcc.append(accuracy(testX, testY, wp))

    plt.plot(np.array(list(range(n))), np.array(trainAcc), '.', label='Train Accuracy', linewidth=0.001)
    plt.plot(np.array(list(range(n))), np.array(testAcc), '.', label='Test Accuracy', linewidth=0.001)
    plt.title('Accuracy, $\\alpha = $' + str(alpha) + ', $\\lambda = $' + str(lam))
    plt.legend()
    plt.ylabel('Accuracy')
    plt.xlabel('Iterations')
    plt.show()


w = np.ones(20)

# Plot 1 - alpha = 0.1

# plotgradascent(X, y, w, 0.1, 10000)

# Plot 2 - alpha = 0.01

# plotgradascent(X, y, w, 0.01, 10000)

# Plot 3 - alpha = 0.001

# plotgradascent(X, y, w, 0.001, 2000)

