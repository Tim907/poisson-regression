import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import load_iris

from efficient_poisson_regression import leverage_score_sampling, uniform_sampling
from efficient_poisson_regression.sampling import (
    ReservoirSampler,
    _round_up,
    compute_leverage_scores,
    compute_leverage_scores_online,
    gibbs_sampler_probit,
    online_ridge_leverage_score_sampling,
    truncated_normal,
    truncated_normal_rejection,
)


def test_uniform_sampling_invalid_shapes():
    X = np.zeros((5, 3))
    y = np.zeros(4)

    with pytest.raises(ValueError):
        uniform_sampling(X, y, sample_size=3)


def test_uniform_sampling_invalid_sample_size():
    X = np.zeros((5, 3))
    y = np.zeros(5)

    with pytest.raises(ValueError):
        uniform_sampling(X, y, sample_size=6)

    with pytest.raises(ValueError):
        uniform_sampling(X, y, sample_size=0)


def test_uniform_sampling():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([1, 2, 4, 5])

    X_sampled, y_sampled = uniform_sampling(X, y, sample_size=3)

    assert X_sampled.shape == (3, 3)
    assert y_sampled.shape == (3,)

    # check if all but one row is in the sample
    rows_in_sample = np.full(4, fill_value=False, dtype=np.bool_)
    for i in range(len(rows_in_sample)):
        cur_row = X[i]
        cur_y = y[i]
        for j in range(3):
            if np.array_equal(X_sampled[j], cur_row) and np.array_equal(
                y_sampled[j], cur_y
            ):
                rows_in_sample[i] = True
                continue

    assert np.sum(rows_in_sample) == 3


def test_leverage_score_sampling():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([1, 2, 4, 5])

    X_sampled, y_sampled, weights = leverage_score_sampling(X, y, sample_size=3)

    assert X_sampled.shape == (3, 3)
    assert y_sampled.shape == (3,)

    # check if all but one row is in the sample
    rows_in_sample = np.full(4, fill_value=False, dtype=np.bool_)
    for i in range(len(rows_in_sample)):
        cur_row = X[i]
        cur_y = y[i]
        for j in range(3):
            if np.array_equal(X_sampled[j], cur_row) and np.array_equal(
                y_sampled[j], cur_y
            ):
                rows_in_sample[i] = True
                continue

    assert np.sum(rows_in_sample) == 3


def test_leverage_score_sampling_augmented():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([1, 2, 4, 5])

    X_sampled, y_sampled, weights = leverage_score_sampling(
        X, y, sample_size=3, augmented=True
    )

    assert X_sampled.shape == (3, 3)
    assert y_sampled.shape == (3,)

    # check if all but one row is in the sample
    rows_in_sample = np.full(4, fill_value=False, dtype=np.bool_)
    for i in range(len(rows_in_sample)):
        cur_row = X[i]
        cur_y = y[i]
        for j in range(3):
            if np.array_equal(X_sampled[j], cur_row) and np.array_equal(
                y_sampled[j], cur_y
            ):
                rows_in_sample[i] = True
                continue

    assert np.sum(rows_in_sample) == 3


def test_compute_leverage_scores_invalid():
    X = np.array([1, 2, 3])

    with pytest.raises(ValueError):
        compute_leverage_scores(X)


def test_compute_leverage_scores():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    true_leverage_scores = np.array(
        [
            0.9977175621816176,
            0.7787312305078449,
            0.36405804147365234,
            0.8594931658368866,
        ]
    )

    leverage_scores = compute_leverage_scores(X)

    assert leverage_scores.shape == (4,)
    assert_allclose(leverage_scores, true_leverage_scores)


@pytest.mark.parametrize("method", ["pinv", "solve"])
def test_compute_leverage_scores_online(method):
    X = np.array([[13, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    true_leverage_scores = compute_leverage_scores(X)

    online_leverage_scores = compute_leverage_scores_online(X, method=method)

    assert online_leverage_scores.shape == (4,)

    # test that online leverage scores are upper bounds
    assert np.all(online_leverage_scores >= true_leverage_scores)


@pytest.mark.parametrize("method", ["pinv", "solve"])
def test_compute_leverage_scores_online_iris(method):
    X, _ = load_iris(return_X_y=True)

    true_leverage_scores = compute_leverage_scores(X)

    online_leverage_scores = compute_leverage_scores_online(X, method=method)

    print(true_leverage_scores)
    print(online_leverage_scores)
    print(online_leverage_scores >= true_leverage_scores)

    assert online_leverage_scores.shape == (150,)

    # test that online leverage scores are upper bounds,
    # epsilon due to numerical reasons
    epsilon = 1e-6
    assert np.all(online_leverage_scores >= true_leverage_scores - epsilon)


def test_online_leverage_scores_invalid_method():
    with pytest.raises(ValueError):
        compute_leverage_scores_online(
            np.array([[1, 2], [3, 4], [5, 7]]), method="something-invalid"
        )


def test_leverage_scores_indifferent_of_labeling():
    """
    In theory, the leverage scores also depend on the labeling vector y.
    This test confirms, that y doesn't matter and it suffices to compute the
    leverage scores only for X.
    """
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    y = np.array([1, -1, 1, -1])

    # multiply the rows of X by labels in y
    Z = y[:, np.newaxis] * X

    leverage_scores_X = compute_leverage_scores(X)
    leverage_scores_Z = compute_leverage_scores(Z)

    assert_allclose(leverage_scores_X, leverage_scores_Z)


def test_reservoir_sampler():
    X, y = load_iris(return_X_y=True)

    # test sampling the entire dataset
    sampler = ReservoirSampler(sample_size=X.shape[0], d=X.shape[1])
    insert_count = 0
    for i in range(X.shape[0]):
        sampler.insert_record(row=X[i], label=y[i], weight=1)
        if sampler.was_last_record_sampled():
            insert_count += 1

    X_sample, y_sample = sampler.get_sample()

    assert_array_equal(X_sample, X)
    assert_array_equal(y_sample, y)
    assert insert_count == X.shape[0]

    # test sampling only one sample
    sampler = ReservoirSampler(sample_size=X.shape[0], d=X.shape[1])
    sampler.insert_record(row=X[0], label=y[0], weight=1)
    X_sample, y_sample = sampler.get_sample()
    assert_array_equal(X_sample[0], X[0])
    assert_array_equal(y_sample, y[0])
    assert sampler.was_last_record_sampled()

    # test sampling only a fraction
    sample_size = 10
    sampler = ReservoirSampler(sample_size=sample_size, d=X.shape[1])
    insert_count = 0
    for i in range(X.shape[0]):
        sampler.insert_record(row=X[i], label=y[i], weight=1)
        if sampler.was_last_record_sampled():
            insert_count += 1

    X_sample, y_sample = sampler.get_sample()
    assert X_sample.shape == (10, X.shape[1])
    assert y_sample.shape == (10,)
    assert insert_count >= sample_size

    # check that the rows and labels in the sample are also in the dataset
    for i in range(sample_size):
        print(i)
        cur_row = X_sample[i]
        cur_label = y_sample[i]
        in_dataset = False
        for j in range(X.shape[0]):
            if np.array_equal(cur_row, X[j]) and np.array_equal(cur_label, y[j]):
                in_dataset = True
                break
        assert in_dataset


def test_online_ridge_leverage_score_sampling():
    X, y = load_iris(return_X_y=True)

    sample_size = 10
    X_sample, y_sample, w_sample = online_ridge_leverage_score_sampling(
        X=X, y=y, sample_size=sample_size, augmentation_constant=1 / X.shape[0]
    )

    assert X_sample.shape == (sample_size, X.shape[1])
    assert y_sample.shape == (sample_size,)
    assert w_sample.shape == (sample_size,)


def test_round_up():
    x = np.array([0.1, 0.2, 0.25, 0.3, 0.4, 0.6, 0.8, 1, 2, 3, 5, 0, 0.0009765624])
    expected = np.array(
        [0.125, 0.25, 0.25, 0.5, 0.5, 1, 1, 1, 2, 4, 8, 0, 0.0009765625]
    )

    x_rounded = _round_up(x)

    assert_array_equal(x_rounded, expected)

    with pytest.raises(ValueError):
        _round_up(np.array([1, 2, -1, 3]))


def test_truncated_normal():
    result = truncated_normal(a=0, b=np.inf, mean=-3, std=10, size=100, random_state=1)
    assert np.all(result >= 0)

    result = truncated_normal(a=-np.inf, b=0, mean=0, std=10, size=100, random_state=1)
    assert np.all(result <= 0)

    result = truncated_normal(a=1, b=2, mean=1.5, std=10, size=100, random_state=1)
    assert np.all((result >= 1) & (result <= 2))

    a = np.array([1, 10, 1])
    b = np.array([2, 11, 2])
    result = truncated_normal(a, b, mean=0, std=1, size=3)

    assert np.all((result >= a) & (result <= b))


def test_truncated_normal_rejection():
    result = truncated_normal_rejection(a=0, b=np.inf, mean=-3, std=10, size=100)
    assert np.all(result >= 0)

    result = truncated_normal_rejection(a=-np.inf, b=0, mean=0, std=10, size=100)
    assert np.all(result <= 0)

    result = truncated_normal_rejection(a=1, b=2, mean=3, std=1, size=100)
    assert np.all((result > 1) & (result < 2))


def test_gibbs_sampler_probit():
    X, y = load_iris(return_X_y=True)
    y = np.where(y == 1, 1, -1)
    d = X.shape[1]

    # check sample dimension of single chain
    sample = gibbs_sampler_probit(
        X,
        y,
        prior_mean=np.zeros(d),
        prior_cov=np.eye(d),
        num_samples=500,
        num_chains=1,
    )

    assert sample.shape == (500, d)

    # check sample dimension of multiple chains
    sample = gibbs_sampler_probit(
        X,
        y,
        prior_mean=np.zeros(d),
        prior_cov=np.eye(d),
        num_samples=500,
        num_chains=4,
    )

    assert sample.shape == (2000, d)

    # now use the maximum likelihood estimate as a prior and check that
    # the posterior mean is close to the maximum likelihood estimate
    ml_estimate = np.array([0.42271932, -1.11532803, 0.64249527, -1.71604354])
    sample = gibbs_sampler_probit(
        X,
        y,
        prior_mean=ml_estimate,
        prior_cov=np.eye(d),
        num_samples=500,
        num_chains=8,
    )

    assert sample.shape == (4000, d)

    sample_mean = np.mean(sample, axis=0)

    print(ml_estimate)
    print(sample_mean)

    assert_allclose(sample_mean, ml_estimate, rtol=0, atol=0.15)
