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
from consensus.ensemble import CoAssocEnsemble, VotingEnsemble
from consensus.fca import FCAConsensus
from consensus.monti_paper import MontiPaperConsensus
from consensus.runner import ConsensusRunner


@pytest.fixture
def blobs():
    X, y = make_blobs(n_samples=80, centers=3, cluster_std=0.6, random_state=0)
    return X, y


@pytest.fixture
def simple_alg_callables():
    return [
        lambda X: DBSCANWrapper(eps=0.5, min_samples=3).fit_predict(X),
        lambda X: HDBSCANWrapper(min_cluster_size=5).fit_predict(X),
        lambda X: DPCWrapper(percent=2.0).fit_predict(X),
    ]


@pytest.fixture
def label_matrix_3runs(blobs):
    X, _ = blobs
    n = len(X)
    label_matrix = []
    for seed in [0, 1, 2]:
        rng = np.random.RandomState(seed)
        perm = rng.permutation(3)
        labels = np.zeros(n, dtype=int)
        chunk = n // 3
        for c in range(3):
            labels[c*chunk:(c+1)*chunk] = perm[c]
        label_matrix.append(labels)
    return np.array(label_matrix)


def _check_labels(labels, n):
    assert isinstance(labels, np.ndarray)
    assert labels.shape == (n,)
    assert labels.dtype in (np.int32, np.int64, int)


class TestCoAssocEnsemble:
    def test_fit_predict(self, blobs, simple_alg_callables):
        X, _ = blobs
        ens = CoAssocEnsemble(algorithms=simple_alg_callables, n_clusters=3)
        labels = ens.fit(X).labels_
        _check_labels(labels, len(X))
        assert len(set(labels.tolist())) >= 1

    def test_coassoc_matrix_shape(self, blobs, simple_alg_callables):
        X, _ = blobs
        ens = CoAssocEnsemble(algorithms=simple_alg_callables)
        ens.fit(X)
        assert ens.coassoc_matrix_.shape == (len(X), len(X))

    def test_no_oracle_k_required(self, blobs, simple_alg_callables):
        X, _ = blobs
        ens = CoAssocEnsemble(algorithms=simple_alg_callables)
        ens.fit(X)
        assert ens.labels_ is not None


class TestVotingEnsemble:
    def test_fit_predict(self, blobs, simple_alg_callables):
        X, _ = blobs
        ens = VotingEnsemble(algorithms=simple_alg_callables, n_clusters=3)
        labels = ens.fit(X).labels_
        _check_labels(labels, len(X))

    def test_default_k_selection(self, blobs, simple_alg_callables):
        X, _ = blobs
        ens = VotingEnsemble(algorithms=simple_alg_callables)
        ens.fit(X)
        assert ens.labels_ is not None


class TestFCAConsensus:
    def test_basic(self, blobs, simple_alg_callables):
        X, _ = blobs
        fca = FCAConsensus(base_algorithms=simple_alg_callables, noise_label=-1)
        fca.fit(X)
        assert fca.labels_ is not None
        _check_labels(fca.labels_, len(X))


class TestMontiPaper:
    def test_single_alg(self, blobs):
        X, _ = blobs
        fn = lambda Z: DBSCANWrapper(eps=0.5, min_samples=3).fit_predict(Z)
        monti = MontiPaperConsensus(
            base_algorithm=fn,
            n_resamples=5,
            p_sample=0.7,
            random_state=42,
        )
        monti.fit(X)
        assert monti.labels_ is not None
        _check_labels(monti.labels_, len(X))

    def test_builtin_auto_style_callable(self, blobs):
        X, _ = blobs
        from consensus.monti_helpers import builtin_fit_predict_callable

        fn = builtin_fit_predict_callable("dbscan", X, None)
        monti = MontiPaperConsensus(base_algorithm=fn, n_resamples=5, random_state=0)
        monti.fit(X)
        _check_labels(monti.labels_, len(X))

    def test_k_not_oracle(self, blobs):
        X, _ = blobs
        fn = lambda Z: DBSCANWrapper().fit_predict(Z)
        monti = MontiPaperConsensus(base_algorithm=fn, n_resamples=5, random_state=1)
        monti.fit(X)
        assert monti.optimal_k_ is not None and monti.optimal_k_ >= 1


class TestConsensusRunner:
    def test_runner_runs(self, blobs):
        X, _ = blobs
        runner = ConsensusRunner(
            algorithm_names=["dbscan", "hdbscan"],
            consensus_methods=["coassoc", "voting"],
            n_bootstrap=3,
            noise_label=-1,
            random_state=42,
        )
        result = runner.fit(X)
        assert result is not None
        assert hasattr(result, "label_matrix")

    def test_monti2_only_without_ensemble_base(self, blobs):
        X, _ = blobs
        from consensus.monti_helpers import builtin_fit_predict_callable

        fp = builtin_fit_predict_callable("hdbscan", X, None)
        runner = ConsensusRunner(
            algorithm_names=[],
            consensus_methods=["monti2"],
            n_bootstrap=5,
            random_state=0,
            monti2_base_callable=fp,
        )
        r = runner.fit(X)
        assert "monti2" in r.labels
        assert r.monti2_coassoc_matrix is not None
        assert r.coassoc_matrix is None
        assert r.label_matrix is None or r.labels["monti2"].shape[0] == len(X)

    def test_no_k_oracle_in_runner(self):
        runner = ConsensusRunner(algorithm_names=['dbscan'])
        assert runner.n_clusters is None

    def test_label_matrix_shape(self, blobs):
        X, _ = blobs
        runner = ConsensusRunner(
            algorithm_names=['dbscan', 'dpc'],
            consensus_methods=['coassoc'],
            n_bootstrap=2,
            random_state=0,
        )
        result = runner.fit(X)
        assert result.label_matrix.shape[1] == len(X)

    def test_cohirf_single_base_without_ensemble_multibase(self, blobs):
        X, _ = blobs
        from consensus.monti_helpers import builtin_fit_predict_callable

        fp = builtin_fit_predict_callable("hdbscan", X, None)
        runner = ConsensusRunner(
            algorithm_names=[],
            consensus_methods=["cohirf"],
            random_state=0,
            cohirf_base_callable=fp,
        )
        r = runner.fit(X)
        assert "cohirf" in r.labels
        assert r.labels["cohirf"].shape[0] == len(X)
