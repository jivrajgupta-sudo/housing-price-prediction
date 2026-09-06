import numpy as np
from sklearn.utils.estimator_checks import check_estimator

from ta_lib.custom import SignedLog1pTransformer


def test_signed_log1p_round_trip():
    values = np.array([[-100.0, -1.0, 0.0, 2.0, 100.0]])
    transformer = SignedLog1pTransformer().fit(values)

    np.testing.assert_allclose(transformer.inverse_transform(transformer.transform(values)), values)


def test_signed_log1p_passes_sklearn_estimator_checks():
    check_estimator(SignedLog1pTransformer())