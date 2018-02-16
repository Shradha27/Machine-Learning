import matplotlib.pyplot as plt
from matplotlib import style
from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1, 2], [2, 3], [12, 12], [8, 9], [10, 11], [10, 10], [1, 3]])

clf = KMeans(n_clusters=2)
clf.fit(X)

centroids = clf.cluster_centers_
labels = clf.labels_

for i in range(len(centroids)):
    print("centroids:", centroids[i][0], centroids[i][1])

for i in range(len(labels)):
    print(labels[i])

colors = ["r.", "b.", "k.", "g.", "c."]
for i in range(len(X)):
    plt.plot(X[i][0],X[i][1], colors[labels[i]], markersize = 15)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=100, linewidths=5)
plt._show()

