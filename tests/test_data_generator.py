import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_generator.habr_synthetic import all_habr_datasets
from data_generator.uci_real import load_iris_fc, load_wine_fc, load_seeds_fc


class TestHabrDatasets:
    def test_all_returned(self):
        datasets = all_habr_datasets()
        assert len(datasets) >= 5

    def test_shape_consistency(self):
        for ds in all_habr_datasets():
            assert ds.X.ndim == 2, f"{ds.name}: X must be 2D"
            assert ds.y_true.ndim == 1, f"{ds.name}: y_true must be 1D"
            assert ds.X.shape[0] == ds.y_true.shape[0], f"{ds.name}: n_samples mismatch"

    def test_at_least_2_clusters(self):
        for ds in all_habr_datasets():
            k = len(np.unique(ds.y_true))
            assert k >= 2, f"{ds.name}: need at least 2 clusters, got {k}"

    def test_no_nan_in_X(self):
        for ds in all_habr_datasets():
            assert not np.isnan(ds.X).any(), f"{ds.name}: X contains NaN"

    def test_labels_are_int(self):
        for ds in all_habr_datasets():
            assert np.issubdtype(ds.y_true.dtype, np.integer), \
                f"{ds.name}: y_true should be integer, got {ds.y_true.dtype}"


class TestUCILoaders:
    def test_iris_shape(self):
        ds = load_iris_fc()
        assert ds.X.shape == (150, 4)
        assert ds.y_true.shape == (150,)
        assert len(np.unique(ds.y_true)) == 3

    def test_wine_shape(self):
        ds = load_wine_fc()
        assert ds.X.shape == (178, 13)
        assert len(np.unique(ds.y_true)) == 3

    def test_seeds_shape(self):
        ds = load_seeds_fc()
        assert ds.X.shape[0] == 210
        assert len(np.unique(ds.y_true)) == 3

    def test_no_nan(self):
        for loader in [load_iris_fc, load_wine_fc, load_seeds_fc]:
            ds = loader()
            assert not np.isnan(ds.X).any(), f"{ds.name} X has NaN"
            assert not np.isnan(ds.y_true.astype(float)).any(), f"{ds.name} y_true has NaN"

    def test_labels_0indexed(self):
        for loader in [load_iris_fc, load_wine_fc, load_seeds_fc]:
            ds = loader()
            assert ds.y_true.min() >= 0, f"{ds.name} labels should start at 0"
