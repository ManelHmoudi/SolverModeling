import numpy as np

from QINSGA3.algorithm import _normalise_F, _normalise_stats


def test_normalise_F_matches_normalise_stats_composition():
    """_normalise_F(F) must equal (F - ideal) / denom for the ideal/denom
    _normalise_stats(F) returns -- guards the refactor that splits them apart."""
    F = np.array([[0.0, 2.0], [1.0, 0.0], [2.0, 1.0]])

    ideal, denom = _normalise_stats(F)
    expected = (F - ideal) / denom

    np.testing.assert_allclose(_normalise_F(F), expected)
