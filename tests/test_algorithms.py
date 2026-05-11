import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_blobs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algorithms.dbscan_wrapper import DBSCANWrapper
from algorithms.hdbscan_wrapper import HDBSCANWrapper
from algorithms.dpc_wrapper import DPCWrapper
from algorithms.rd_dac_wrapper import RDDACWrapper
from algorithms.ckdpc_wrapper import CKDPCWrapper


@pytest.fixture
def blobs_2d():
    X, y = make_blobs(n_samples=120, centers=3, cluster_std=0.5, random_state=42)
    return X, y


@pytest.fixture
def blobs_high_dim():
    X, y = make_blobs(n_samples=80, centers=3, n_features=8, cluster_std=0.6, random_state=7)
    return X, y


def _check_labels(labels, n):
    assert isinstance(labels, np.ndarray), "labels must be ndarray"
    assert labels.shape == (n,), f"Expected shape ({n},), got {labels.shape}"
    assert labels.dtype in (np.int32, np.int64, int), f"labels dtype should be int-like, got {labels.dtype}"


def _k_found(labels):
    return len(set(labels[labels != -1].tolist()))


class TestDBSCAN:
    def test_output_shape(self, blobs_2d):
        X, _ = blobs_2d
        labels = DBSCANWrapper(eps=0.5, min_samples=3).fit_predict(X)
        _check_labels(labels, len(X))

    def test_finds_clusters(self, blobs_2d):
        X, _ = blobs_2d
        labels = DBSCANWrapper(eps=0.5, min_samples=3).fit_predict(X)
        assert _k_found(labels) >= 2

    def test_auto_eps(self, blobs_2d):
        X, _ = blobs_2d
        labels = DBSCANWrapper().fit_predict(X)
        _check_labels(labels, len(X))

    def test_no_k_passed(self, blobs_2d):
        X, _ = blobs_2d
        alg = DBSCANWrapper(eps=0.5, min_samples=3)
        assert not hasattr(alg, 'n_clusters') or alg.n_clusters is None  # no oracle k

    def test_high_dim(self, blobs_high_dim):
        X, _ = blobs_high_dim
        labels = DBSCANWrapper().fit_predict(X)
        _check_labels(labels, len(X))


class TestHDBSCAN:
    def test_output_shape(self, blobs_2d):
        X, _ = blobs_2d
        labels = HDBSCANWrapper(min_cluster_size=5).fit_predict(X)
        _check_labels(labels, len(X))

    def test_finds_clusters(self, blobs_2d):
        X, _ = blobs_2d
        labels = HDBSCANWrapper(min_cluster_size=5).fit_predict(X)
        assert _k_found(labels) >= 2

    def test_default_params(self, blobs_2d):
        X, _ = blobs_2d
        labels = HDBSCANWrapper().fit_predict(X)
        _check_labels(labels, len(X))


class TestDPC:
    def test_output_shape(self, blobs_2d):
        X, _ = blobs_2d
        labels = DPCWrapper(percent=2.0).fit_predict(X)
        _check_labels(labels, len(X))

    def test_finds_clusters(self, blobs_2d):
        X, _ = blobs_2d
        labels = DPCWrapper(percent=2.0).fit_predict(X)
        assert _k_found(labels) >= 1

    def test_no_n_clusters_oracle(self, blobs_2d):
        X, _ = blobs_2d
        alg = DPCWrapper()
        assert alg.n_clusters is None

    def test_default_runs(self, blobs_2d):
        X, _ = blobs_2d
        labels = DPCWrapper().fit_predict(X)
        _check_labels(labels, len(X))


class TestRDDAC:
    def test_output_shape(self, blobs_2d):
        X, _ = blobs_2d
        labels = RDDACWrapper(k=7).fit_predict(X)
        _check_labels(labels, len(X))

    def test_finds_clusters(self, blobs_2d):
        X, _ = blobs_2d
        labels = RDDACWrapper(k=7).fit_predict(X)
        assert _k_found(labels) >= 1

    def test_no_n_clusters_oracle(self):
        alg = RDDACWrapper()
        assert alg.n_clusters is None

    def test_default_runs(self, blobs_2d):
        X, _ = blobs_2d
        labels = RDDACWrapper().fit_predict(X)
        _check_labels(labels, len(X))


class TestCKDPC:
    def test_output_shape(self, blobs_2d):
        X, _ = blobs_2d
        labels = CKDPCWrapper(alpha=0.5, percent=2.0).fit_predict(X)
        _check_labels(labels, len(X))

    def test_finds_clusters(self, blobs_2d):
        X, _ = blobs_2d
        labels = CKDPCWrapper(alpha=0.5, percent=2.0).fit_predict(X)
        assert _k_found(labels) >= 1

    def test_no_n_clusters_oracle(self):
        alg = CKDPCWrapper()
        assert alg.n_clusters is None

    def test_default_runs(self, blobs_2d):
        X, _ = blobs_2d
        labels = CKDPCWrapper().fit_predict(X)
        _check_labels(labels, len(X))
