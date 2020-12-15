import pandas as pd
import numpy as np
import random
import matplotlib

df_raw = pd.read_csv('../Datasets/dataset.csv', header=0)
X = np.array(df_raw)

init_centroids = random.sample(range(0, len(df_raw)), 10)

centroids = []
for i in init_centroids:
    centroids.append(df_raw.loc[i])
centroids = np.array(centroids)

# print(len(centroids))

def calc_distance(X1, X2):
    return(sum((X1 - X2)**2))**0.5

def findClosestCentroids(ic, X):
    assigned_centroid = []
    for i in X:
        distance=[]
        for j in ic:
            distance.append(calc_distance(i, j))
        assigned_centroid.append(np.argmin(distance))
    return assigned_centroid

get_centroids = findClosestCentroids(centroids, X)


def calc_centroids(clusters, X):
    new_centroids = []
    new_df = pd.concat([pd.DataFrame(X), pd.DataFrame(clusters, columns=['cluster'])],
                      axis=1)
    print(new_df.head())
    for c in set(new_df['cluster']):
        current_cluster = new_df[new_df['cluster'] == c][new_df.columns[:-1]]
        print(current_cluster.head())
        cluster_mean = current_cluster.mean(axis=0)
        new_centroids.append(cluster_mean)
    return new_centroids

for i in range(1):
    get_centroids = findClosestCentroids(centroids, X)
    centroids = calc_centroids(get_centroids, X)

# print(centroids)