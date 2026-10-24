import numpy as np
import pytest
from validation_benchmark import file_holdout, score_split, speed_holdouts


def test_grouped_holdout_keeps_overlapping_windows_together():
    y = np.repeat(["normal", "normal", "fault", "fault"], 10)
    groups = np.repeat(["n20", "n30", "f20", "f30"], 10)
    train, test = file_holdout(y, groups)
    assert not set(groups[train]) & set(groups[test])
    assert set(y[train]) == set(y[test]) == {"normal", "fault"}
    assert len(train) + len(test) == len(y)


def test_insufficient_files_never_falls_back_to_random_windows():
    with pytest.raises(ValueError, match="two independent"):
        file_holdout([0, 0, 1, 1], ["a", "a", "b", "b"])


def test_speed_holdout_is_source_disjoint():
    y = np.tile([0, 1], 4)
    groups = np.repeat(["a", "b", "c", "d"], 2)
    speeds = np.repeat([20, 20, 30, 30], 2)
    for _, train, test in speed_holdouts(y, groups, speeds):
        assert not set(groups[train]) & set(groups[test])
        assert not set(speeds[train]) & set(speeds[test])


def test_unknown_speed_fails_closed():
    with pytest.raises(ValueError, match="known speeds"):
        list(speed_holdouts([0, 1], ["a", "b"], [0, 30]))


def test_nonfinite_speed_is_not_a_new_regime():
    with pytest.raises(ValueError, match="known speeds"):
        list(speed_holdouts([0, 1], ["a", "b"], [np.nan, 30]))


def test_speed_holdout_requires_class_coverage_in_both_parts():
    with pytest.raises(ValueError, match="class coverage"):
        list(speed_holdouts([0, 1, 0], ["a", "b", "c"], [20, 20, 30]))


def test_protected_scoring_refuses_source_overlap():
    with pytest.raises(ValueError, match="Source file crosses"):
        score_split(np.ones((4, 2)), np.array([0, 1, 0, 1]), np.array(["a", "b", "a", "b"]),
                    np.array([0, 1]), np.array([2, 3]))


@pytest.mark.parametrize("labels,groups", [([], []), ([0, 1], ["a"])])
def test_group_inputs_must_be_aligned_and_nonempty(labels, groups):
    with pytest.raises(ValueError, match="aligned"):
        file_holdout(labels, groups)
