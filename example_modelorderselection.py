import numpy as np
from skstab import ModelOrderSelection
from skstab.datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

dataset = 'exemples2_5g'
X, y = load_dataset(dataset)
print('Dataset: {} (true number of clusters: K = {})'.format(dataset, len(np.unique(y))))

algorithm = KMeans
km_kwargs = {'init': 'k-means++', 'n_init': 10}

k_values = list(range(2, 11))
print('Evaluated numbers of clusters:', k_values)

stab = ModelOrderSelection(X, algorithm,
                           param_name='n_clusters',
                           param_values=k_values,
                           classifier=KNeighborsClassifier,
                           norm_samples=20,
                           runs=20,
                           algo_kwargs=km_kwargs,
                           clf_kwargs={'n_neighbors': 1},
                           n_jobs=-1)

score = stab.score()
print('Model order selection scores:\n', score)
k_hat = stab.select_param()[0]
print('Selected number of clusters: K =', k_hat)
