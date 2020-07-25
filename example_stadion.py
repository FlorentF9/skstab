import os
import matplotlib.pyplot as plt
import numpy as np
from skstab import StadionEstimator
from skstab.datasets import load_dataset
from sklearn.cluster import KMeans

dataset = 'exemples2_5g'
X, y = load_dataset(dataset)
print('Dataset: {} (true number of clusters: K = {})'.format(dataset, len(np.unique(y))))

algorithm = KMeans
km_kwargs = {'init': 'k-means++', 'n_init': 10}

k_values = list(range(1, 11))
omega = list(range(2, 6))
print('Evaluated numbers of clusters:', k_values)

stab = StadionEstimator(X, algorithm,
                        param_name='n_clusters',
                        param_values=k_values,
                        omega=omega,
                        extended=True,
                        runs=10,
                        perturbation='uniform',
                        perturbation_kwargs='auto',
                        algo_kwargs=km_kwargs,
                        n_jobs=-1)

score = stab.score(strategy='max', crossing=True)
print('Stadion-max scores:\n', score)
k_hat = stab.select_param()[0]
print('Selected number of clusters: K =', k_hat)

"""
Generate plots
"""
# These are the "Tableau 20" colors as RGB.
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)
epsilons = [kwargs['eps'] for kwargs in stab.perturbation_kwargs]
bstab_path = 'stabB_paths_{}_{}.pdf'.format(algorithm.__name__, dataset)
wstab_path = 'stabW_paths_{}_{}.pdf'.format(algorithm.__name__, dataset)
stadion_path = 'stadion_paths_{}_{}.pdf'.format(algorithm.__name__, dataset)

plt.figure(figsize=(10, 8))
for i, p in enumerate(k_values):
    plt.plot(epsilons, stab.between_cluster_stability_paths[i, :, 0], label='{} = {}'.format(stab.param_name, p),
             color=tableau20[(2 * i) % 20])
plt.legend()
plt.xlabel('$\epsilon$', fontsize=16)
plt.ylabel('Between-cluster Stability', fontsize=16)
plt.title('{} (dataset = {} / true K = {})'.format(algorithm.__name__, dataset, len(np.unique(y))), fontsize=16)
plt.savefig(bstab_path, bbox_inches='tight')

plt.figure(figsize=(10, 8))
for i, p in enumerate(k_values):
    plt.plot(epsilons, stab.within_cluster_stability_paths[i, :, 0], label='{} = {}'.format(stab.param_name, p),
             color=tableau20[(2 * i) % 20])
plt.legend()
plt.xlabel('$\epsilon$', fontsize=16)
plt.ylabel('Within-cluster Stability', fontsize=16)
plt.title('{} (dataset = {} / true K = {})'.format(algorithm.__name__, dataset, len(np.unique(y))), fontsize=16)
plt.savefig(wstab_path, bbox_inches='tight')

plt.figure(figsize=(10, 8))
for i, p in enumerate(k_values):
    plt.plot(epsilons, stab.stadion_paths[i, :, 0], label='{} = {}'.format(stab.param_name, p),
             color=tableau20[(2 * i) % 20])
plt.legend()
plt.xlabel('$\epsilon$', fontsize=16)
plt.ylabel('Stadion', fontsize=16)
plt.title('{} (dataset = {} / true K = {})'.format(algorithm.__name__, dataset, len(np.unique(y))), fontsize=16)
plt.savefig(stadion_path, bbox_inches='tight')
print('Paths saved to {}, {} and {}!'.format(bstab_path, wstab_path, stadion_path))
