import numpy as np
import scipy
import torch
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier


def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise ValueError(f"Unknown type: {type(tensor)}")


def feature_transform(Z: np.ndarray, u: np.ndarray, d_spu: int = 1, scale: float = 0) -> np.ndarray:
    scales = np.ones(Z.shape[1])
    scales[:d_spu] = scale
    # print(Z.shape, u.shape, scales.shape)
    Z = Z @ u @ np.diag(scales)
    return Z


def check_labels(labels) -> int:
    classes = np.unique(labels)
    n_classes = len(classes)
    assert np.all(classes == np.arange(n_classes)), f"Labels must be 0, 1, 2, ..., {n_classes - 1}"
    return n_classes


def estimate_means(zs, ys, gs, n_envs, n_classes) -> dict:
    Zs, Ys = {}, {}
    for e in range(n_envs):
        Zs[e] = zs[gs == e]
        Ys[e] = ys[gs == e]
    Mus = {}
    for label in range(n_classes):
        means = {}
        for e in range(n_envs):
            means[e] = np.mean(Zs[e][Ys[e] == label], axis=0)
        Mus[label] = np.vstack(list(means.values()))
    return Mus


def estimate_covs(zs, ys, gs, n_envs, n_classes) -> dict:
    Zs, Ys = {}, {}
    for e in range(n_envs):
        Zs[e] = zs[gs == e]
        Ys[e] = ys[gs == e]
    Covs = {label: {} for label in range(n_classes)}
    for label in range(n_classes):
        for e in range(n_envs):
            Covs[label][e] = np.cov(Zs[e][Ys[e] == label].T)
    return Covs


def check_clf(clf, n_classes):
    if isinstance(clf, LogisticRegression) or isinstance(clf, RidgeClassifier) or isinstance(clf, SGDClassifier):
        if n_classes == 2:
            assert 1 <= clf.coef_.shape[0] <= 2, f"The output dim of a binary classifier must be 1 or 2"
        else:
            assert clf.coef_.shape[0] == n_classes, f"The output dimension of the classifier must be {n_classes}."
        return clf
    elif isinstance(clf, torch.nn.Linear):
        weight = clf.weight.detach().data.cpu().numpy()
        bias = clf.bias.detach().data.cpu().numpy()
    elif isinstance(clf, dict):
        weight, bias = to_numpy(clf['weight']), to_numpy(clf['bias'])
    else:
        raise ValueError(f"Unknown classifier type: {type(clf)}")

    assert weight.shape[0] == len(
        bias), f"The output dimension of weight should match bias: {weight.shape[0]} vs {len(bias)}"
    sklearn_clf = LogisticRegression()
    sklearn_clf.n_classes = n_classes
    sklearn_clf.classes_ = np.arange(n_classes)
    sklearn_clf.coef_ = weight
    sklearn_clf.intercept_ = bias
    assert sklearn_clf.coef_.shape[0] == n_classes, f"The output dimension of the classifier must be {n_classes}."

    return sklearn_clf


class ISRClassifier:
    default_clf_kwargs = dict(C=1, max_iter=1000, random_state=0)

    def __init__(self, version: str = 'mean', pca_dim: int = -1, d_spu: int = -1, spu_scale: float = 0,
                 chosen_class=None, clf_type: str = 'LogisticRegression', clf_kwargs: dict = None,
                 ):
        self.version = version

        self.pca_dim = pca_dim
        self.d_spu = d_spu
        self.spu_scale = spu_scale

        self.clf_kwargs = ISRClassifier.default_clf_kwargs if clf_kwargs is None else clf_kwargs
        self.clf_type = clf_type
        self.chosen_class = chosen_class
        self.Us = {}  # stores computed projection matrices
        assert self.clf_type in ['LogisticRegression', 'RidgeClassifier', 'SGDClassifier'], \
            f"Unknown classifier type: {self.clf_type}"

    def set_params(self, **params):
        for name, val in params.items():
            setattr(self, name, val)

    def fit(self, features, labels, envs, chosen_class: int = None, d_spu: int = None, given_clf=None,
            spu_scale: float = None):

        # estimate the stats (mean & cov) and fit a PCA if requested
        self.fit_data(features, labels, envs)

        if chosen_class is None:
            assert self.chosen_class is not None, "chosen_class must be specified if not given in the constructor"
            chosen_class = self.chosen_class

        if self.version == 'mean':
            self.fit_isr_mean(chosen_class=chosen_class, d_spu=d_spu)
        elif self.version == 'cov':
            self.fit_isr_cov(chosen_class=chosen_class, d_spu=d_spu)
        else:
            raise ValueError(f"Unknown ISR version: {self.version}")

        self.fit_clf(features, labels, given_clf=given_clf, spu_scale=spu_scale)
        return self

    def fit_data(self, features, labels, envs, n_classes=None, n_envs=None):
        # estimate the mean and covariance of each class per environment
        self.n_classes = check_labels(labels)
        self.n_envs = check_labels(envs)
        if n_classes is not None: assert self.n_classes == n_classes
        if n_envs is not None: assert self.n_envs == n_envs

        # fit a PCA if requested
        if self.pca_dim > 0:
            self.pca = PCA(n_components=self.pca_dim).fit(features)
            features = self.pca.transform(features)
        else:
            self.pca = None
        self.means = estimate_means(features, labels, envs, self.n_envs, self.n_classes)
        self.covs = estimate_covs(features, labels, envs, self.n_envs, self.n_classes)
        return features

    def fit_isr_mean(self, chosen_class: int, d_spu: int = None):
        d_spu = self.d_spu if d_spu is None else d_spu
        assert d_spu < self.n_envs
        assert 0 <= chosen_class < self.n_classes
        # We project features into a subspace, and d_spu is the dimension of the subspace
        # Wew derive theoretically in the paper that the projection dimension of ISR-Mean
        # is at most n_envs-1
        if d_spu <= 0: self.d_spu = self.n_envs - 1

        key = ('mean', chosen_class, self.d_spu)
        if key in self.Us:
            return self.Us[key]

        # Estimate the empirical mean of each class

        # This PCA is just a helper function to obtain the projection matrix
        helper_pca = PCA(n_components=self.d_spu).fit(self.means[chosen_class])
        # The projection matrix has dimension (orig_dim, d_spu)
        # The SVD is just to pad the projection matrix with columns (the dimensions orthogonal
        # to the projection subspace) that makes the matrix a full-rank square matrix.
        U_proj = helper_pca.components_.T

        self.U = np.linalg.qr(U_proj, mode='complete')[0].real
        # The first d_spu dimensions of U correspond to spurious features, which we will
        # discard or reduce. The remaining dimensions are of the invariant feature subspace that
        # the algorithm identifies (not necessarily to be the real invariant features).

        # If we want to discard the spurious features, we can simply reduce the first d_spu
        # dimensions of U to zeros. However, this may hurt the performance of the algorithm sometimes,
        # so we can use the following strategy: rescale of the first d_spu dimensions with
        # factor between 0 and 1. This rescale factor is spu_scale that is chosen by the user.
        # print('\neig(U):', np.real(np.linalg.eigvals(self.U)))
        # print('singular vals:', s)
        self.Us[key] = self.U
        return self.U

    def fit_isr_cov(self, chosen_class: int, d_spu: int = None):
        self.d_spu = d_spu if d_spu is not None else self.d_spu
        assert self.d_spu > 0, "d_spu must be provided for ISR-Cov"
        # TODO: implement ISR-Cov for n_envs > 2
        assert self.n_envs == 2, "ISR-Cov is only implemented for binary env so far"

        key = ('cov', chosen_class, self.d_spu)
        if key in self.Us:
            return self.Us[key]

        env_pair = [0, 1]
        cov_0 = self.covs[chosen_class][env_pair[0]]
        cov_1 = self.covs[chosen_class][env_pair[1]]
        cov_diff = cov_1 - cov_0
        D = cov_diff.shape[0]

        # take square root of cov_diff such that the resulting matrix has non-negative eigenvalues
        # the largest d_spu eigenvalues correspond to the spurious feature subspace
        # we only need compute the eigenvectors of these d_spu dimensions (save computation cost)

        cov_sqr = cov_diff @ cov_diff
        w, U_proj = scipy.linalg.eigh(cov_sqr, subset_by_index=[D - self.d_spu, D - 1])
        assert w.min() >= 0
        order = np.flip(np.argsort(w).flatten())
        U_proj = U_proj[:, order]

        # trivially call SVD to fill the rest columns the (D-d_spu)-dim subspace orthogonal to the
        # spurious feature subspace

        self.U = np.linalg.svd(U_proj, full_matrices=True)[0]

        self.Us[key] = self.U

        return self.U

    def fit_clf(self, features=None, labels=None, given_clf=None, sample_weight=None):
        if given_clf is None:
            assert features is not None and labels is not None
            self.clf = getattr(linear_model, self.clf_type)(**self.clf_kwargs)
            features = self.transform(features, )
            self.clf.fit(features, labels, sample_weight=sample_weight)
        else:
            self.clf = check_clf(given_clf, n_classes=self.n_classes)
            self.clf.coef_ = self.clf.coef_ @ self.U
        return self.clf

    def transform(self, features, ):
        if self.pca is not None:
            features = self.pca.transform(features)
        new_zs = feature_transform(features, u=self.U,
                                   d_spu=self.d_spu, scale=self.spu_scale)
        return new_zs

    def predict(self, features):
        zs = self.transform(features)
        return self.clf.predict(zs)

    def score(self, features, labels):
        zs = self.transform(features)
        return self.clf.score(zs, labels)

    def fit_transform(self, features, labels, envs, chosen_class, given_clf=None):
        self.fit(features, labels, envs, chosen_class, given_clf)
        return self.transform(features)
