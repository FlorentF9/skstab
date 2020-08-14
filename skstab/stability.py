"""
skstab - Stability estimators

@author Florent Forest, Alex Mourer
"""

from skstab.perturbation import *
from skstab.metrics import minimum_matching_distance
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_array
from joblib import Parallel, delayed

EPS = 1e-12


def _perturbation_call(perturbation):
    """Get perturbation callable"""
    if isinstance(perturbation, str) and perturbation == 'uniform':
        perturbation_call = uniform_additive_noise
    elif isinstance(perturbation, str) and perturbation == 'gaussian':
        perturbation_call = gaussian_additive_noise
    elif isinstance(perturbation, str) and perturbation == 'shift':
        perturbation_call = random_shift
    elif isinstance(perturbation, str) and perturbation == 'offset':
        perturbation_call = random_offset
    elif isinstance(perturbation, str) and perturbation == 'scale':
        perturbation_call = random_scale
    elif isinstance(perturbation, str) and perturbation == 'warp':
        perturbation_call = random_warp
    elif isinstance(perturbation, str) and perturbation == 'subsample':
        perturbation_call = subsample
    elif isinstance(perturbation, str) and perturbation == 'bootstrap':
        perturbation_call = bootstrap
    elif callable(perturbation):
        perturbation_call = perturbation
    else:
        raise ValueError("the perturbation parameter should be 'uniform', 'gaussian', "
                         "'shift', 'offset', 'scale', 'warp', 'subsample', 'bootstrap' "
                         "or a callable, '{}' (type '{}') was passed.".format(perturbation, type(perturbation)))
    return perturbation_call


def _check_measure(measure):
    """Check (dis)similarity measure"""
    if callable(measure):
        measure_call_list = [measure]
    elif isinstance(measure, list) and all(callable(m) for m in measure):
        measure_call_list = measure
    else:
        raise ValueError("the measure parameter should be a callable or a list of callable similarity "
                         "measure, '{}' (type '{}') was passed.".format(measure, type(measure)))
    return measure_call_list


def _check_similarity(similarity, length):
    """Check if measures are similarities or dissimilarities"""
    if isinstance(similarity, bool):
        similarity_list = [similarity] * length
    elif isinstance(similarity, list) and all(isinstance(s, bool) for s in similarity) and len(similarity) == length:
        similarity_list = similarity
    else:
        raise ValueError("the similarity parameter should be a boolean or a list of booleans indicating "
                         "if measures are similarities or dissimilarities, '{}' (type '{}') "
                         "was passed.".format(similarity, type(similarity)))
    return similarity_list


def _get_labels(model, X):
    """Get labels from a trained sklearn-like model"""
    try:
        y = model.labels_
    except AttributeError:
        y = model.predict(X)
    return y


class BaseStability:
    """Base class for stability analysis.

    Parameters
    ----------
    X : array
        data matrix.
    algorithm : object
        algorithm class with a sklearn-like API.
    param_name : str
        name of algorithm parameter determining the number of clusters.
    param_values : list
        list of parameter values to evaluate.
    perturbation : str or callable
        operation used to generate perturbed versions of the data set.
    measure : callable or list<callable>
        similarity or dissimilarity function between two clusterings.
    similarity : bool or list<bool>
        boolean indicating whether each measure is a similarity (True) or a dissimilarity (False). Must have the same
        length as the measure parameter.
    runs : int
        number of perturbed samples.
    perturbation_kwargs : list<dict> (default = [{}])
        keyword arguments to pass to the perturbation function.
    algo_kwargs : dict (default = {})
        keyword arguments to pass to the algorithm class.
    n_jobs : int (default = -1)
        number of parallel processes.

    Returns
    -------
    self
        BaseStability object.
    """

    def __init__(self, X, algorithm, param_name, param_values, perturbation, measure, similarity, runs,
                 perturbation_kwargs=[{}], algo_kwargs={}, n_jobs=-1):
        assert(hasattr(X, '__array__'))
        self.X = check_array(X, dtype=X.dtype.type, copy=True)
        self.algorithm = algorithm
        self.param_name = param_name
        self.param_values = param_values
        self.perturbation = _perturbation_call(perturbation)
        self.measure = _check_measure(measure)
        self.similarity = _check_similarity(similarity, len(self.measure))
        self.runs = runs
        self.perturbation_kwargs = perturbation_kwargs
        self.algo_kwargs = algo_kwargs
        self.n_jobs = n_jobs
        return self


class ReferenceComparisonStability(BaseStability):
    """Stability estimation by comparison with a reference partition.

    Parameters
    ----------
    X : array
        data matrix.
    algorithm : object
        algorithm class with a sklearn-like API.
    param_name : str
        name of algorithm parameter determining the number of clusters.
    param_values : list
        list of parameter values to evaluate.
    extended : bool
        use extension operator to extend cluster labels to the perturbed data sets.
    perturbation : str or callable
        operation used to generate perturbed versions of the data set.
    measure : callable or list<callable>
        similarity or dissimilarity function between two clusterings.
    similarity : bool or list<bool>
        boolean indicating whether each measure is a similarity (True) or a dissimilarity (False). Must have the same
        length as the measure parameter.
    runs : int
        number of perturbed samples.
    perturbation_kwargs : list<dict> (default = [{}])
        keyword arguments to pass to the perturbation function.
    algo_kwargs : dict (default = {})
        keyword arguments to pass to the algorithm class.
    n_jobs : int (default = -1)
        number of parallel processes.

    Returns
    -------
    self
        ReferenceComparisonStability object.
    """

    def __init__(self, X, algorithm, param_name, param_values, extended, perturbation, measure, similarity, runs,
                 perturbation_kwargs=[{}], algo_kwargs={}, n_jobs=-1):
        self.extended = extended
        self.y_ref = None  # reference partitions
        if self.extended:
            self.model_ref = None  # reference models for extended version
        super().__init__(X, algorithm, param_name, param_values, perturbation, measure, similarity, runs,
                         perturbation_kwargs, algo_kwargs, n_jobs)
        self.compute_reference_partitions()

    def compute_reference_partitions(self):
        self.y_ref = {}
        if self.extended:
            self.model_ref = {}
        for p in self.param_values:
            model = self.algorithm(**{self.param_name: p}, **self.algo_kwargs).fit(self.X)
            self.y_ref[p] = _get_labels(model, self.X)
            if self.extended:
                self.model_ref[p] = model

    def _stability(self, param_value, perturbation_kwargs={}):
        """Core method to compute a stability score"""
        x_perturbed = self.perturbation(self.X, **perturbation_kwargs)
        if self.extended:
            model = self.model_ref[param_value]
            y = model.predict(x_perturbed)
        else:
            model = self.algorithm(**{self.param_name: param_value}, **self.algo_kwargs).fit(x_perturbed)
            y = _get_labels(model, x_perturbed)
        return np.array([s(y, self.y_ref[param_value]) for s in self.measure])


class PairwiseComparisonStability(BaseStability):
    """Stability estimation by pairwise comparison between perturbed data sets.

    Parameters
    ----------
    X : array
        data matrix.
    algorithm : object
        algorithm class with a sklearn-like API.
    param_name : str
        name of algorithm parameter determining the number of clusters.
    param_values : list
        list of parameter values to evaluate.
    perturbation : str or callable
        operation used to generate perturbed versions of the data set.
    measure : callable or list<callable>
        similarity or dissimilarity function between two clusterings.
    similarity : bool or list<bool>
        boolean indicating whether each measure is a similarity (True) or a dissimilarity (False). Must have the same
        length as the measure parameter.
    runs : int
        number of perturbed samples.
    perturbation_kwargs : list<dict> (default = [{}])
        keyword arguments to pass to the perturbation function.
    algo_kwargs : dict (default = {})
        keyword arguments to pass to the algorithm class.
    n_jobs : int (default = -1)
        number of parallel processes.

    Returns
    -------
    self
        PairwiseComparisonStability object.
    """

    def __init__(self, X, algorithm, param_name, param_values, perturbation, measure, similarity, runs,
                 perturbation_kwargs=[{}], algo_kwargs={}, n_jobs=-1):
        super().__init__(X, algorithm, param_name, param_values, perturbation, measure, similarity, runs,
                         perturbation_kwargs, algo_kwargs, n_jobs)

    def _stability(self, param_value, perturbation_kwargs={}):
        subsample1, subidx1 = self.perturbation(self.X, **perturbation_kwargs)
        subsample2, subidx2 = self.perturbation(self.X, **perturbation_kwargs)
        model1 = self.algorithm(**{self.param_name: param_value}, **self.algo_kwargs).fit(subsample1)
        model2 = self.algorithm(**{self.param_name: param_value}, **self.algo_kwargs).fit(subsample2)
        y1 = _get_labels(model1, subsample1)
        y2 = _get_labels(model2, subsample2)
        _, idx1, idx2 = np.intersect1d(subidx1, subidx2, return_indices=True)
        return np.array([s(y1[idx1], y2[idx2]) for s in self.measure])


class LabelTransferStability(BaseStability):
    """Stability estimation by transferring labels from one data set half to the other using a supervised classifier.

    Parameters
    ----------
    X : array
        data matrix.
    algorithm : object
        algorithm class with a sklearn-like API.
    param_name : str
        name of algorithm parameter determining the number of clusters.
    param_values : list
        list of parameter values to evaluate.
    classifier : object
        classification algorithm class with a sklearn-like API.
    measure : callable or list<callable>
        similarity or dissimilarity function between two clusterings.
    similarity : bool or list<bool>
        boolean indicating whether each measure is a similarity (True) or a dissimilarity (False). Must have the same
        length as the measure parameter.
    runs : int
        number of perturbed samples.
    algo_kwargs : dict (default = {})
        keyword arguments to pass to the algorithm class.
    n_jobs : int (default = -1)
        number of parallel processes.

    Returns
    -------
    self
        LabelTransferStability object.
    """

    def __init__(self, X, algorithm, param_name, param_values, classifier, measure, similarity, runs,
                 algo_kwargs={}, clf_kwargs={}, n_jobs=-1):
        self.classifier = classifier
        self.clf_kwargs = clf_kwargs
        super().__init__(X, algorithm, param_name, param_values, train_test_split, measure, similarity, runs, [{}],
                         algo_kwargs, n_jobs)

    def _stability(self, param_value):
        split1, split2 = train_test_split(self.X, test_size=0.5)
        model1 = self.algorithm(**{self.param_name: param_value}, **self.algo_kwargs).fit(split1)
        model2 = self.algorithm(**{self.param_name: param_value}, **self.algo_kwargs).fit(split2)
        y1 = _get_labels(model1, split1)
        y2 = _get_labels(model2, split2)
        y2_clf = self.classifier(**self.clf_kwargs).fit(split1, y1).predict(split2)
        return np.array([s(y2, y2_clf) for s in self.measure])


class StadionEstimator(ReferenceComparisonStability):
    """Stadion (stability difference criterion) from Mourer et al, 2020.

    Parameters
    ----------
    X : array
        data matrix.
    algorithm : object
        algorithm class with a sklearn-like API.
    param_name : str
        name of algorithm parameter determining the number of clusters.
    param_values : list
        list of parameter values to evaluate.
    omega : list
        list of parameter values for within-cluster stability estimation.
    extended : bool (default = False)
        use extension operator to extend cluster labels to the perturbed data sets.
    perturbation : str or callable (default = 'uniform')
        operation used to generate perturbed versions of the data set.
    measure : callable or list<callable> (default = adjusted_rand_score)
        similarity or dissimilarity function between two clusterings.
    similarity : bool or list<bool> (default = True)
        boolean indicating whether each measure is a similarity (True) or a dissimilarity (False). Must have the same
        length as the measure parameter.
    runs : int (default = 10)
        number of perturbed samples.
    perturbation_kwargs : list<dict> or 'auto' (default = 'auto')
        keyword arguments to pass to the perturbation function. If set to 'auto', the 'eps' parameter takes 10
        linearly spaced values in the [0, sqrt(X.shape[1])] interval.
    algo_kwargs : dict (default = {})
        keyword arguments to pass to the algorithm class.
    n_jobs : int (default = -1)
        number of parallel processes.

    Returns
    -------
    self
        StadionEstimator object.

    References
    ----------
    [1] Mourer, A., Forest, F., Lebbah, M., Azzag, H., & Lacaille, J. (2020). Selecting the Number of Clusters K with a
    Stability Trade-off: an Internal Validation Criterion. https://arxiv.org/abs/2006.08530
    """

    def __init__(self, X, algorithm, param_name, param_values, omega, extended=False, perturbation='uniform',
                 measure=adjusted_rand_score, similarity=True, runs=10, perturbation_kwargs='auto', algo_kwargs={},
                 n_jobs=-1):
        self.omega = omega
        self.between_cluster_stability_paths_ = None
        self.within_cluster_stability_paths_ = None
        self.stadion_paths_ = None
        if isinstance(perturbation_kwargs, str) and perturbation_kwargs == 'auto':
            epsmax = np.sqrt(X.shape[1])
            perturbation_kwargs = [{'eps': eps} for eps in np.linspace(0.0, epsmax, num=10)]
        elif not isinstance(perturbation_kwargs, list) or not len(perturbation_kwargs) > 0 \
                or not all(isinstance(kwargs, dict) for kwargs in perturbation_kwargs):
            raise ValueError("the perturbation_kwargs parameter should be 'auto' (uniform noise only) "
                             "or a non-empty list of dict, '{}' (type '{}') "
                             "was passed.".format(perturbation_kwargs, type(perturbation_kwargs)))
        super().__init__(X, algorithm, param_name, param_values, extended, perturbation, measure, similarity, runs,
                         perturbation_kwargs, algo_kwargs, n_jobs)

    def stability_path(self, param_value):
        """Compute stability path for a given parameter and reference partition on perturbed versions of the input.

        Parameters
        ----------
        param_value
            algorithm parameter value determining the number of clusters.

        Returns
        -------
        stab_path : array of shape (n_perturbation, n_measure)
            stability score averaged over runs, for each perturbation_kwargs and measure.
        """
        stab_path = np.zeros((len(self.perturbation_kwargs), len(self.measure), self.runs))
        for i, kwargs in enumerate(self.perturbation_kwargs):
            for run in range(self.runs):
                stab_path[i, :, run] = self._stability(param_value, kwargs)
        return stab_path.mean(axis=-1)

    @property
    def between_cluster_stability_paths(self):
        """Between-cluster stability paths for each parameter (lazy-evaluated property).

        Returns
        -------
        between_cluster_stability_paths_ : array of shape (n_param_values, n_perturbation, n_measure)
            between-cluster stability score averaged over runs, for each parameter, perturbation_kwargs and measure.
        """
        if self.between_cluster_stability_paths_ is None:
            if self.n_jobs == 1:
                stab_paths = np.zeros((len(self.param_values), len(self.perturbation_kwargs), len(self.measure)))
                for i, p in enumerate(self.param_values):
                    stab_paths[i] = self.stability_path(p)
            else:
                def stability_job(i):
                    return self.stability_path(self.param_values[i])
                stab_paths = np.array(
                    Parallel(n_jobs=self.n_jobs)(delayed(stability_job)(i) for i in range(len(self.param_values)))
                )
            self.between_cluster_stability_paths_ = stab_paths
        return self.between_cluster_stability_paths_

    @property
    def within_cluster_stability_paths(self):
        """Within-cluster stability paths for each parameter (lazy-evaluated property). Instantiates a recursive
        StadionEstimator on each cluster of the reference partition, whenever the cluster size is larger than the
        number of clusters in omega.

        Returns
        -------
        within_cluster_stability_paths_ : array of shape (n_param_values, n_perturbation, n_measure)
            within-cluster stability score averaged over runs and parameters in omega, for each parameter,
            perturbation_kwargs and measure.
        """
        if self.within_cluster_stability_paths_ is None:
            def instability_job(i):
                wstab_path = np.zeros((len(self.perturbation_kwargs), len(self.measure)))
                for c in np.unique(self.y_ref[self.param_values[i]]):  # for each cluster c
                    cluster = self.X[self.y_ref[self.param_values[i]] == c]
                    weight = cluster.shape[0] / self.X.shape[0]
                    omega_valid = [p for p in self.omega if p < cluster.shape[0]]  # parameters with a valid clustering
                    if len(omega_valid) > 0:
                        estimator = StadionEstimator(cluster, self.algorithm, self.param_name, omega_valid, None,
                                                     self.extended, self.perturbation, self.measure, self.similarity,
                                                     self.runs, self.perturbation_kwargs, self.algo_kwargs, n_jobs=1)
                        # stab_paths = self._compute_stability_paths(cluster, params_within, cluster_y_ref, n_jobs=1)
                        stab_paths = estimator.between_cluster_stability_paths
                        del estimator
                        wstab_path += np.mean(stab_paths, axis=0) * weight
                    else:
                        wstab_path += 1.0 * weight
                return wstab_path
            self.within_cluster_stability_paths_ = np.array(
                Parallel(n_jobs=self.n_jobs)(delayed(instability_job)(i) for i in range(len(self.param_values)))
            )
        return self.within_cluster_stability_paths_

    @property
    def stadion_paths(self):
        """Stadion (stability difference criterion) paths for each parameter (lazy-evaluated property).

        Returns
        -------
        stadion_paths_ : array of shape (n_param_values, n_perturbation, n_measure)
            stadion paths, equal to the difference between between-cluster and within-cluster stability paths.
        """
        if self.stadion_paths_ is None:
            self.stadion_paths_ = self.between_cluster_stability_paths - self.within_cluster_stability_paths
        return self.stadion_paths_

    @staticmethod
    def _find_crossing(stadion_paths, param_values, y_ref):
        """Find crossing point until which all paths are under the path with a single cluster (K=1).

        Parameters
        ----------
        stadion_paths : array of shape (n_param_values, n_perturbation, n_measure)
            stadion paths, as computed by the StadionEstimator.stadion_paths method.
        param_values : list
            list of parameter values to evaluate.
        y_ref : array of shape (n_param_values, n_samples)
            reference clustering for each parameter.

        Returns
        -------
        limit : int
            index of perturbation until which all paths are smaller than the path corresponding to K=1.
        """
        k1_idx = np.array([i for i in range(len(param_values))
                           if (y_ref[param_values[i]] == 0).all()])  # indices where K=1
        if k1_idx.size == 0:
            raise ValueError("no path corresponding to a single cluster (K=1)!")
        score_ = np.delete(stadion_paths, k1_idx, axis=0)
        k1 = k1_idx[0]  # k1 corresponds to the first index where K=1
        crossed = (score_ <= stadion_paths[k1] + EPS).all(axis=0)  # scores are under the K=1 paths
        if not crossed.any():
            raise ValueError("score paths are always greater than the K=1 path. Increase perturbation level!")
        limit = crossed.size + 1
        for l in range(crossed.size - 1, -1, -1):  # go until all paths have crossed K=1
            if not crossed[l]:
                limit = l + 1
                break
        return limit

    def score(self, strategy='max', crossing=True):
        """Aggregate stadion path using specified strategy, until crossing with K=1 solution (optional).

        Parameters
        ----------
        strategy : string or callable (default = 'max')
            stadion path aggregation strategy ('max', 'mean' or callable).
        crossing : bool (default = True
            aggregate only until all paths are under the path with a single cluster (K=1).

        Returns
        -------
        score : array of shape (n_param_values, n_measure)
            stadion score for each parameter and measure.
        """
        stadion = self.stadion_paths
        if isinstance(crossing, bool) and crossing:
            crossing = StadionEstimator._find_crossing(stadion, self.param_values, self.y_ref)
        elif isinstance(crossing, bool):
            crossing = stadion.shape[1]
        else:
            raise ValueError("the crossing parameter should be a boolean (True or False), "
                             "'{}' (type '{}') was passed.".format(crossing, type(crossing)))
        if isinstance(strategy, str) and strategy == 'max':
            score = np.max(stadion[:, :crossing], axis=1)
        elif isinstance(strategy, str) and strategy in ['mean', 'avg', 'average']:
            score = np.mean(stadion[:, :crossing], axis=1)
        elif callable(strategy):
            score = np.apply_along_axis(strategy, axis=1, arr=stadion[:, :crossing])
        else:
            raise ValueError("the strategy parameter should be 'max', 'mean' or a callable, "
                             "'{}' (type '{}') was passed.".format(strategy, type(strategy)))
        return score

    def select_param(self, strategy='max', crossing=True):
        """Select optimal parameter among param_values by taking the highest aggregated stadion score.

        Parameters
        ----------
        strategy : string or callable (default = 'max')
            stadion path aggregation strategy ('max', 'mean' or callable).
        crossing : bool (default = True
            aggregate only until all paths are under the path with a single cluster (K=1).

        Returns
        -------
        params : list
            parameters selected by the stadion method, for each similarity measure.
        """
        score = self.score(strategy, crossing)
        for i in range(len(self.measure)):
            if not self.similarity[i]:
                score[:, i] = -score[:, i]
        return [self.param_values[k] for k in score.argmax(axis=0)]


class ModelExplorer(PairwiseComparisonStability):
    """Model explorer algorithm from Ben-Hur et al, 2002.

    Parameters
    ----------
    X : array
        data matrix.
    algorithm : object
        algorithm class with a sklearn-like API.
    param_name : str
        name of algorithm parameter determining the number of clusters.
    param_values : list
        list of parameter values to evaluate.
    f : float (default = 0.8)
        subsample fraction of input data set.
    measure : callable or list<callable> (default = fowlkes_mallows_score)
        similarity or dissimilarity function between two clusterings.
    similarity : bool or list<bool> (default = True)
        boolean indicating whether each measure is a similarity (True) or a dissimilarity (False). Must have the same
        length as the measure parameter.
    runs : int (default = 100)
        number of perturbed samples.
    algo_kwargs : dict (default = {})
        keyword arguments to pass to the algorithm class.
    n_jobs : int (default = -1)
        number of parallel processes.

    Returns
    -------
    self
        ModelExplorer object.

    References
    ----------
    [1] Ben-Hur, A., Elisseeff, A., & Guyon, I. (2002). A stability based method for discovering structure in clustered
    data. Pacific Symposium on Biocomputing. https://doi.org/10.1142/9789812799623_0002
    """

    def __init__(self, X, algorithm, param_name, param_values, f=0.8, measure=fowlkes_mallows_score, similarity=True,
                 runs=100, algo_kwargs={}, n_jobs=-1):
        self.score_ = None
        super().__init__(X, algorithm, param_name, param_values, 'subsample', measure, similarity, runs,
                         [{'f': f, 'return_indices': True}], algo_kwargs, n_jobs)

    def stability(self, param_value):
        """Compute stability score for a given parameter on input subsample. As specified in [1], the score
        corresponds to P(s > 0.9), where s are the similarity values.

        Parameters
        ----------
        param_value
            algorithm parameter value determining the number of clusters.

        Returns
        -------
        stab : array of shape (n_measure,)
            stability score averaged over runs, for each measure.
        """
        stab = np.zeros((len(self.measure), self.runs))
        for run in range(self.runs):
            stab[:, run] = self._stability(param_value, self.perturbation_kwargs[0])
        return (stab > 0.9).mean(axis=-1)  # P(stab) > 0.9 as in Ben-Hur et al, 2002)

    def score(self):
        """Model explorer scores.

        Returns
        -------
        score : array of shape (n_param_values, n_measure)
            model explorer score for each parameter and measure.
        """
        if self.score_ is None:
            if self.n_jobs == 1:
                stab = np.zeros((len(self.param_values), len(self.perturbation_kwargs), len(self.measure)))
                for i, p in enumerate(self.param_values):
                    stab[i] = self.stability(p)
            else:
                def stability_job(i):
                    return self.stability(self.param_values[i])
                stab = np.array(
                    Parallel(n_jobs=self.n_jobs)(delayed(stability_job)(i) for i in range(len(self.param_values)))
                )
            self.score_ = stab
        return self.score_

    def select_param(self):
        """Select optimal parameter among param_values by taking the maximum jump in stability scores.

        Returns
        -------
        params : list
            parameters selected by the model explorer method, for each similarity measure.
        """
        score = self.score()
        for i in range(len(self.measure)):
            if not self.similarity[i]:
                score[:, i] = -score[:, i]
        return [self.param_values[k] for k in np.diff(score, 1, axis=0).argmin(axis=0)]


class ModelOrderSelection(LabelTransferStability):
    """Model order selection method from Lange et al, 2004.

    Parameters
    ----------
    X : array
        data matrix.
    algorithm : object
        algorithm class with a sklearn-like API.
    param_name : str
        name of algorithm parameter determining the number of clusters.
    param_values : list
        list of parameter values to evaluate.
    classifier : object (default = KNeighborsClassifier)
        classification algorithm class with a sklearn-like API.
    norm_samples : int (default = 20)
        number of samples for random label normalization.
    measure : callable or list<callable> (default = minimum_matching_distance)
        similarity or dissimilarity function between two clusterings.
    similarity : bool or list<bool> (default = False)
        boolean indicating whether each measure is a similarity (True) or a dissimilarity (False). Must have the same
        length as the measure parameter.
    runs : int (default = 20)
        number of perturbed samples.
    algo_kwargs : dict (default = {})
        keyword arguments to pass to the algorithm class.
    clf_kwargs : dict (default = {})
        keyword arguments to pass to the classifier class.
    n_jobs : int (default = -1)
        number of parallel processes.

    Returns
    -------
    self
        ModelOrderSelection object.

    References
    ----------
    [1] Lange, T., Roth, V., Braun, M. L., & Buhmann, J. M. (2004). Stability-based validation of clustering solutions.
    Neural Computation. https://doi.org/10.1162/089976604773717621
    """

    def __init__(self, X, algorithm, param_name, param_values, classifier=KNeighborsClassifier, norm_samples=20,
                 measure=minimum_matching_distance, similarity=False, runs=20, algo_kwargs={}, clf_kwargs={},
                 n_jobs=-1):
        self.norm_samples = norm_samples
        self.score_ = None
        super().__init__(X, algorithm, param_name, param_values, classifier, measure, similarity, runs, algo_kwargs,
                         clf_kwargs, n_jobs)

    def stability(self, param_value):
        """Compute stability for a given parameter with label transfer and random sample normalization.

        Parameters
        ----------
        param_value
            algorithm parameter value determining the number of clusters.

        Returns
        -------
        stab : array of shape (n_measure,)
            stability score normalized and averaged over runs, for each measure.
        """
        stab = np.zeros((len(self.measure), self.runs))
        for run in range(self.runs):
            stab[:, run] = self._stability(param_value)
        norm = np.zeros((len(self.measure), self.norm_samples))
        for i in range(self.norm_samples):
            y_rand1 = np.random.choice(param_value, size=self.X.shape[0])
            y_rand2 = np.random.choice(param_value, size=self.X.shape[0])
            norm[:, i] = np.array([s(y_rand1, y_rand2) for s in self.measure])
        stab = stab / norm.mean(axis=1, keepdims=True)
        return stab.mean(axis=-1)

    def score(self):
        """Model order selection score.

        Returns
        -------
        score : array of shape (n_param_values, n_measure)
            model explorer score for each parameter and measure.
        """
        if self.score_ is None:
            if self.n_jobs == 1:
                stab = np.zeros((len(self.param_values), len(self.perturbation_kwargs), len(self.measure)))
                for i, p in enumerate(self.param_values):
                    stab[i] = self.stability(p)
            else:
                def stability_job(i):
                    return self.stability(self.param_values[i])
                stab = np.array(
                    Parallel(n_jobs=self.n_jobs)(delayed(stability_job)(i) for i in range(len(self.param_values)))
                )
            self.score_ = stab
        return self.score_

    def select_param(self):
        """Select optimal parameter among param_values by taking the highest stability score.

        Returns
        -------
        params : list
            parameters selected by the model order selection method, for each similarity measure.
        """
        score = self.score()
        for i in range(len(self.measure)):
            if not self.similarity[i]:
                score[:, i] = -score[:, i]
        return [self.param_values[k] for k in score.argmax(axis=0)]

