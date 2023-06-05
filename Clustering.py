# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 18:55:26 2023

@author: swank
"""

#Clustering with K-means
#Understanding centroid-based algorithms
### DO NOT RUN WITH KMEANS
### DEPRICATED -> MEM LEAK


import numpy as np
A = np.array([165, 55, 70])
B = np.array([185, 60, 30])
​
D = (A - B)
D = D**2
D = np.sqrt(np.sum(D))
​
print(D)

#Creating an example with image data
from sklearn.datasets import load_digits
digits = load_digits()
X = digits.data
ground_truth = digits.target
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
pca = PCA(n_components=30)
Cx = pca.fit_transform(scale(X))
print('Explained variance %0.3f' 
      % sum(pca.explained_variance_ratio_))
from sklearn.cluster import KMeans
clustering = KMeans(n_clusters=10, 
                    n_init=10, random_state=1)
clustering.fit(Cx)

#Looking for optimal solutions
import numpy as np
import pandas as pd
ms = np.column_stack((ground_truth,clustering.labels_))
df = pd.DataFrame(ms, 
                  columns = ['Ground truth','Clusters'])
pd.crosstab(df['Ground truth'], df['Clusters'], 
            margins=True)
import numpy as np
inertia = list()
for k in range(1,21):
    clustering = KMeans(n_clusters=k, 
                        n_init=10, random_state=1)
    clustering.fit(Cx)
    inertia.append(clustering.inertia_)
delta_inertia = np.diff(inertia) * -1

import matplotlib.pyplot as plt
%matplotlib inline
plt.figure()
x_range = [k for k in range(2, 21)]
plt.xticks(x_range)
plt.plot(x_range, delta_inertia, 'ko-')
plt.xlabel('Number of clusters')
plt.ylabel('Rate of change of inertia')
plt.show()

#Clustering big data

###########################################
#C:\ProgramData\anaconda3\lib\site-packages\sklearn\cluster\_kmeans.py:1382: UserWarning: KMeans is known to have a memory leak on Windows with MKL, when there are less chunks than available threads. You can avoid it by setting the environment variable OMP_NUM_THREADS=8.
  #warnings.warn(
      
#############################################

k = 10
clustering = KMeans(n_clusters=k, 
                    n_init=10, random_state=1)
clustering.fit(Cx)
kmeans_inertia = clustering.inertia_
print("K-means inertia: %0.1f" % kmeans_inertia)
from sklearn.cluster import MiniBatchKMeans
batch_clustering = MiniBatchKMeans(n_clusters=k, 
                                   random_state=1)
batch = 100
for row in range(0, len(Cx), batch):
    if row+batch < len(Cx):
        feed = Cx[row:row+batch,:]
    else:
        feed = Cx[row:,:]
    batch_clustering.partial_fit(feed)
batch_inertia = batch_clustering.score(Cx) * -1
​
print("MiniBatchKmeans inertia: %0.1f" % batch_inertia)


#Performing Hierarchical Clustering
#Using a hierarchical cluster solution

from sklearn.cluster import AgglomerativeClustering
​
Hclustering = AgglomerativeClustering(n_clusters=10, 
                               affinity='euclidean', 
                               linkage='ward')
Hclustering.fit(Cx)
​
ms = np.column_stack((ground_truth,Hclustering.labels_))
df = pd.DataFrame(ms, 
                  columns = ['Ground truth','Clusters'])
pd.crosstab(df['Ground truth'], 
            df['Clusters'], margins=True)
Using a two-phase clustering solution
from sklearn.cluster import KMeans
clustering = KMeans(n_clusters=50, 
                    n_init=10,
                    random_state=1)
clustering.fit(Cx)
Kx = clustering.cluster_centers_
Kx_mapping = {case:cluster for case,
   cluster in enumerate(clustering.labels_)}
from sklearn.cluster import AgglomerativeClustering
Hclustering = AgglomerativeClustering(n_clusters=10,
                                      affinity='cosine', 
                                      linkage='complete')
Hclustering.fit(Kx)
H_mapping = {case:cluster for case,
   cluster in enumerate(Hclustering.labels_)}
final_mapping = {case:H_mapping[Kx_mapping[case]]
   for case in Kx_mapping}
ms = np.column_stack((ground_truth,
 [final_mapping[n] for n in range(max(final_mapping)+1)]))
df = pd.DataFrame(ms, 
                  columns = ['Ground truth','Clusters'])
pd.crosstab(df['Ground truth'], 
            df['Clusters'], margins=True)

#Discovering new groups with DBScan
from sklearn.cluster import DBSCAN
DB = DBSCAN(eps=3.7, min_samples=15)
DB.fit(Cx)

from collections import Counter
print('No. clusters: %i' % len(np.unique(DB.labels_))) 
print(Counter(DB.labels_))
​
ms = np.column_stack((ground_truth, DB.labels_))
df = pd.DataFrame(ms, 
                  columns = ['Ground truth', 'Clusters'])
​
pd.crosstab(df['Ground truth'], 
            df['Clusters'], margins=True)
