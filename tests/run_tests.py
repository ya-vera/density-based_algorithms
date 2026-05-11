import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.datasets import make_blobs

from algorithms.dbscan_wrapper import DBSCANWrapper
from algorithms.hdbscan_wrapper import HDBSCANWrapper
from algorithms.dpc_wrapper import DPCWrapper
from algorithms.rd_dac_wrapper import RDDACWrapper
from algorithms.ckdpc_wrapper import CKDPCWrapper
from data_generator.habr_synthetic import all_habr_datasets
from data_generator.uci_real import load_iris_fc, load_wine_fc, load_seeds_fc


PASS = 0
FAIL = 0


def run(name, fn):
    global PASS, FAIL
    try:
        fn()
        print(f"  PASS  {name}")
        PASS += 1
    except Exception:
        print(f"  FAIL  {name}")
        traceback.print_exc()
        FAIL += 1


X_blobs, y_blobs = make_blobs(n_samples=80, centers=3, cluster_std=0.5, random_state=42)


def _check_labels(labels, n):
    assert isinstance(labels, np.ndarray), "not ndarray"
    assert labels.shape == (n,), f"bad shape {labels.shape}"


def _k_found(labels):
    return len(set(labels[labels != -1].tolist()))


print("\n── Algorithm tests ─────────────────────────────────────────")

for name, cls, kwargs in [
    ("DBSCAN default",    DBSCANWrapper,  {}),
    ("DBSCAN eps=0.5",    DBSCANWrapper,  {"eps": 0.5, "min_samples": 3}),
    ("HDBSCAN default",   HDBSCANWrapper, {}),
    ("HDBSCAN mcs=5",     HDBSCANWrapper, {"min_cluster_size": 5}),
    ("DPC default",       DPCWrapper,     {}),
    ("DPC percent=2",     DPCWrapper,     {"percent": 2.0}),
    ("RD-DAC default",    RDDACWrapper,   {}),
    ("RD-DAC k=7",        RDDACWrapper,   {"k": 7}),
    ("CKDPC default",     CKDPCWrapper,   {}),
    ("CKDPC alpha=0.5",   CKDPCWrapper,   {"alpha": 0.5, "percent": 2.0}),
]:
    def _make(cls=cls, kwargs=kwargs):
        labels = cls(**kwargs).fit_predict(X_blobs)
        _check_labels(labels, len(X_blobs))
    run(name, _make)

print("\n── No oracle k tests ───────────────────────────────────────")

for name, cls in [("DBSCAN eps=None by default", DBSCANWrapper),
                  ("DPC n_clusters=None", DPCWrapper),
                  ("RD-DAC n_clusters=None", RDDACWrapper),
                  ("CKDPC n_clusters=None", CKDPCWrapper)]:
    def _check(cls=cls, name=name):
        inst = cls()
        if 'n_clusters' in name:
            assert inst.n_clusters is None
        else:
            assert inst.eps is None
    run(name, _check)

print("\n── Data generator tests ─────────────────────────────────────")

def test_habr():
    ds_list = all_habr_datasets()
    assert len(ds_list) >= 5
    for ds in ds_list:
        assert ds.X.ndim == 2
        assert ds.X.shape[0] == ds.y_true.shape[0]
        assert not np.isnan(ds.X).any()
        assert len(np.unique(ds.y_true)) >= 2
run("habr datasets", test_habr)

def test_iris():
    ds = load_iris_fc()
    assert ds.X.shape == (150, 4)
    assert len(np.unique(ds.y_true)) == 3
run("UCI iris", test_iris)

def test_wine():
    ds = load_wine_fc()
    assert ds.X.shape == (178, 13)
    assert len(np.unique(ds.y_true)) == 3
run("UCI wine", test_wine)

def test_seeds():
    ds = load_seeds_fc()
    assert ds.X.shape[0] == 210
    assert len(np.unique(ds.y_true)) == 3
run("UCI seeds", test_seeds)

print("\n── Consensus tests ──────────────────────────────────────────")

from consensus.ensemble import CoAssocEnsemble, VotingEnsemble
from consensus.runner import ConsensusRunner

_alg_callables = [
    lambda X: DBSCANWrapper(eps=0.5, min_samples=3).fit_predict(X),
    lambda X: HDBSCANWrapper(min_cluster_size=5).fit_predict(X),
]

def test_coassoc():
    ens = CoAssocEnsemble(algorithms=_alg_callables, n_clusters=3)
    ens.fit(X_blobs)
    assert ens.labels_ is not None
    _check_labels(ens.labels_, len(X_blobs))
run("CoAssocEnsemble basic", test_coassoc)

def test_voting():
    ens = VotingEnsemble(algorithms=_alg_callables, n_clusters=3)
    ens.fit(X_blobs)
    _check_labels(ens.labels_, len(X_blobs))
run("VotingEnsemble basic", test_voting)

def test_runner():
    runner = ConsensusRunner(
        algorithm_names=['dbscan', 'hdbscan'],
        consensus_methods=['coassoc', 'voting'],
        n_bootstrap=2,
        random_state=0,
    )
    result = runner.fit(X_blobs)
    assert result is not None
    assert result.label_matrix.shape[1] == len(X_blobs)
run("ConsensusRunner coassoc+voting", test_runner)

print(f"\n{'='*50}")
print(f"Results: {PASS} passed, {FAIL} failed")
sys.exit(0 if FAIL == 0 else 1)
