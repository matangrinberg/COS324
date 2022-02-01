import numpy as np
import matplotlib.pyplot as plt
import time

# Importing Data As Shown In http://rasbt.github.io/mlxtend/user_guide/data/loadlocal_mnist/

from mlxtend.data import loadlocal_mnist

(X, y) = loadlocal_mnist(
    images_path='/Users/matangrinberg/PycharmProjects/COS324PS4/train-images-idx3-ubyte',
    labels_path='/Users/matangrinberg/PycharmProjects/COS324PS4/train-labels-idx1-ubyte')

# Processing Data

X_data = np.asarray(np.divide(X, 255))
y_data = y[:, np.newaxis]
X_len = X_data.shape[0]  # 60,000
X_size = X_data.shape[1]  # 784 (28 x 28)


# Establishing Cluster # and Responsibilities

k = 4

resp = np.zeros([X_len, k])
OH_values = np.random.randint(0, k, X_len)
for i in range(0, X_len):
    resp[i, OH_values[i]] = 1


# Defining Functions


def resp_matrix(vec):
    length = vec.shape[0]
    response = np.zeros([length, k])
    for j in range(0, length):
        response[j, vec[j]] = 1

    return response


def cluster_numbers(mat):
    return np.sum(mat, 0)


def k_plusplus_init():
    better_means = np.zeros([X_size, 1])
    seed = np.random.randint(0, X_len)
    better_means[:, 0] = X_data[seed, :]
    for count in range(1, k):
        XtX = np.sum(X_data * X_data, axis=1)[:, np.newaxis]  # 60,000 x 1
        MutMu = np.sum(better_means.transpose() * better_means.transpose(), axis=1)[np.newaxis, :]  # 1 x k
        XtMu = np.dot(X_data, better_means)  # 60,000 x k
        all_dists = ((-2 * XtMu) + XtX) + MutMu
        min_dists = all_dists.min(axis=1)
        probabilities = min_dists / sum(min_dists)
        probabilities = probabilities.clip(min=0)
        r = np.random.choice(np.arange(0, 60000), p=probabilities)
        better_means = np.append(better_means, X_data[r][:, np.newaxis], 1)

    return better_means


def cluster_means(r_mat):    # 784 x k
    inverse_numbers = 1 / (cluster_numbers(r_mat))
    unscaled_means = np.dot(resp.transpose(), X_data)
    return np.multiply(unscaled_means.transpose(), inverse_numbers)


def dist(v1, v2):
    return np.norm(v1, v2)


def objective(resp_mat, mean_mat):
    XtX = np.sum(X_data * X_data, axis=1)[:, np.newaxis]  # 60,000 x 1
    MutMu = np.sum(mean_mat.transpose() * mean_mat.transpose(), axis=1)[np.newaxis, :]  # 1 x k
    XtMu = np.dot(X_data, mean_mat)  # 60,000 x k
    dist = ((-2 * XtMu) + XtX) + MutMu
    return np.sum(np.multiply(dist, resp_mat))


# Employing Lloyd's Algorithm

# obj_list = []
start = time.time()
change = 1
#means = k_plusplus_init()
while change > 0:
    means = cluster_means(resp)
    old_resp = resp
    XtX = np.sum(X_data * X_data, axis=1)[:, np.newaxis]  # 60,000 x 1
    MutMu = np.sum(means.transpose() * means.transpose(), axis=1)[np.newaxis, :]  # 1 x k
    XtMu = np.dot(X_data, means)  # 60,000 x k
    distances = ((-2 * XtMu) + XtX) + MutMu

    new = np.argmin(distances, axis=1)
    resp = resp_matrix(new)
    #means = cluster_means(resp)
    diff = resp - old_resp
    change = np.sum(diff * diff)
    print(change)
    # obj_list.append(np.sum(np.multiply(resp, distances)))

end = time.time()



# plt.plot(obj_list/(np.min(obj_list)))
# plt.xlabel("Iteration")
# plt.ylabel("Objective Function (normalized)")
# plt.show()


# Displaying Images

image_means = means.reshape(28, 28, k)


def means_plot(images=image_means):
    for i in range(0, k):
        plt.subplot(3, 3, i + 1)
        plt.imshow(images[:, :, i])
    plt.show()


