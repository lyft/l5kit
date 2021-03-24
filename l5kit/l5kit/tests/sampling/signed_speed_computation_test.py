import numpy as np

from l5kit.sampling.agent_sampling import compute_signed_speeds


def test_zero_speeds() -> None:
    input = np.zeros([12, 2])
    expected_output = np.zeros([12])
    output = compute_signed_speeds(vels_mps=input)
    assert np.all(output == expected_output)


def test_positive_speeds() -> None:
    input = np.ones([12, 2])
    expected_output = np.ones([12]) * np.sqrt(2)
    output = compute_signed_speeds(vels_mps=input)
    assert np.all(output == expected_output)


def test_negative_speeds() -> None:
    input = -np.ones([12, 2])
    expected_output = -np.ones([12]) * np.sqrt(2)
    output = compute_signed_speeds(vels_mps=input)
    assert np.all(output == expected_output)


def test_mixed_signed_speeds() -> None:
    input = np.array([[1, 1], [1, 1], [-1, -1]])
    expected_output = np.array([1, 1, -1]) * np.sqrt(2)
    output = compute_signed_speeds(vels_mps=input)
    assert np.all(output == expected_output)
