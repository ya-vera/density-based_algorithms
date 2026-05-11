import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation.metrics import ClusteringMetrics


@pytest.fixture
def perfect():
    X = np.random.RandomState(0).randn(9, 2)
    y_pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return X, y_pred, y_true


@pytest.fixture
def noisy():
    X = np.random.RandomState(1).randn(9, 2)
    y_pred = np.array([-1, 0, 0, 1, -1, 1, 2, 2, -1])
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    return X, y_pred, y_true


@pytest.fixture
def single_cluster():
    X = np.random.RandomState(2).randn(6, 2)
    y_pred = np.array([0, 0, 0, 0, 0, 0])
    y_true = np.array([0, 0, 1, 1, 2, 2])
    return X, y_pred, y_true


@pytest.fixture
def all_noise():
    X = np.random.RandomState(3).randn(6, 2)
    y_pred = np.array([-1, -1, -1, -1, -1, -1])
    y_true = np.array([0, 0, 1, 1, 2, 2])
    return X, y_pred, y_true


class TestClusteringMetrics:
    def test_perfect_ari(self, perfect):
        X, y_pred, y_true = perfect
        m = ClusteringMetrics(y_pred, y_true, X)
        ext = m.external()
        assert abs(ext['ari'] - 1.0) < 1e-6

    def test_perfect_nmi(self, perfect):
        X, y_pred, y_true = perfect
        m = ClusteringMetrics(y_pred, y_true, X)
        ext = m.external()
        assert abs(ext['nmi'] - 1.0) < 1e-6

    def test_noise_fraction(self, noisy):
        X, y_pred, y_true = noisy
        m = ClusteringMetrics(y_pred, y_true, X)
        struct = m.structure()
        assert abs(struct['noise_fraction'] - 3/9) < 1e-6

    def test_n_clusters(self, perfect):
        X, y_pred, y_true = perfect
        m = ClusteringMetrics(y_pred, y_true, X)
        struct = m.structure()
        assert struct['n_clusters'] == 3

    def test_single_cluster_external(self, single_cluster):
        X, y_pred, y_true = single_cluster
        m = ClusteringMetrics(y_pred, y_true, X)
        ext = m.external()
        assert ext['ari'] == 0.0 or (isinstance(ext['ari'], float) and ext['ari'] <= 0)

    def test_all_metrics_returns_dict(self, perfect):
        X, y_pred, y_true = perfect
        m = ClusteringMetrics(y_pred, y_true, X)
        result = m.all_metrics()
        assert isinstance(result, dict)
        assert 'ari' in result or 'ARI' in result or len(result) > 0

    def test_internal_metrics_need_X(self, perfect):
        X, y_pred, y_true = perfect
        m = ClusteringMetrics(y_pred, y_true, X)
        internal = m.internal()
        assert isinstance(internal, dict)

    def test_no_y_true_skips_external(self, perfect):
        X, y_pred, _ = perfect
        m = ClusteringMetrics(y_pred, X=X)
        ext = m.external()
        assert ext is None or all(v is None or (isinstance(v, float) and v != v)
                                   for v in (ext or {}).values())
