import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
import pandas as pn
from geopy.distance import great_circle

rawData = pn.read_csv('cities100.csv')
data = np.transpose(rawData.as_matrix())
coordinates = np.transpose(np.array([data[2], data[3]]))
distance_matrix=np.zeros([100, 100])

for i in range(0, 100):
    for j in range(0, 100):
        distance_matrix[i, j] = great_circle(coordinates[i], coordinates[j]).meters

distance_matrix = 1/2 * (distance_matrix + np.transpose(distance_matrix))

# link = hierarchy.linkage(distance_matrix, 'single')
# link = hierarchy.linkage(distance_matrix, 'average')
# link = hierarchy.linkage(distance_matrix, 'weighted')
# link = hierarchy.linkage(distance_matrix, 'median')
# link = hierarchy.linkage(distance_matrix, 'complete')
# link = hierarchy.linkage(distance_matrix, 'centroid')
link = hierarchy.linkage(distance_matrix, 'ward')

plt.figure(dpi=500)
dend = hierarchy.dendrogram(link, labels=data[0], orientation='right')
plt.show()
plt.savefig("graph.png", dpi=500)