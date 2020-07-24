"""
skstab - Data loading utility functions

@author Florent Forest, Alex Mourer
"""

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


ARTIFICIAL_DATASETS = [
    '2d_10c', 
    'r15', 
    'd31', 
    'elliptical_10_2', 
    'twenty',
    'fourty',
    'hepta', 
    'tetra',
    '2d-3c-no123', 
    '2d-4c',
    '2d-4c-no4',
    '2d-4c-no9',
    'curves1',
    'diamond9',
    'ds-577',
    'ds-850',
    'elly-2d10c13s',
    'engytime',
    'ds4c2sc8',
    'long1',
    'long2',
    'long3',
    'longsquare',
    'sizes1',
    'sizes2',
    'sizes3',
    'sizes4',
    'sizes5',
    'spherical_4_3',
    'spherical_5_2',
    'spherical_6_2',
    'square1',
    'square2',
    'square3',
    'square4',
    'square5',
    'st900',
    'triangle1',
    'triangle2',
    'twodiamonds',
    'wingnut',
    'xclara',
    'zelnik2',
    'zelnik4',
    'exemples5_overlap2_3g',
    'exemples4_overlap_3g',
    'exemples8_Overlap_Uvar_5g',
    'exemples9_YoD_6g',
    'exemples6_quicunx_4g',
    'exemples10_WellS_3g',
    'exemples3_Uvar_4g',
    'exemples1_3g',
    'exemples2_5g',
    'exemples7_elbow_3g',
    '3clusters_elephant',
    '4clusters_corner',
    '4clusters_twins',
    '5clusters_stars',
    'a1',
    'a2',
    'g2-2',
    'g2-16',
    'g2-64',
    's1',
    's2',
    's3',
    's4',
]

REAL_DATASETS = [
    'crabs',
    'iris',
    'oldfaithful',
    'wine_umap',
    'mfds_umap',
    'usps_umap',
    'mnist_umap'
]


def load_dataset(name, path='./datasets'):
    if name in ARTIFICIAL_DATASETS:
        data_path = os.path.join(path, 'data/artificial')
    elif name in REAL_DATASETS:
        data_path = os.path.join(path, 'data/real')
    else:
        raise ValueError('Unknown dataset!')
    target_path = os.path.join(path, 'target')
    x = pd.read_csv(os.path.join(data_path, '{}.csv'.format(name)), sep=';', decimal=',').values
    y = pd.read_csv(os.path.join(target_path, '{}.csv'.format(name))).values.ravel()
    x_scaled = StandardScaler().fit_transform(x)
    del x
    return x_scaled, y
