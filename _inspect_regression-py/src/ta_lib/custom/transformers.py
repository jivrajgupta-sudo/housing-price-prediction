"""Custom scikit-learn compatible transformers for the project."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data


class SignedLog1pTransformer(TransformerMixin, BaseEstimator):
    """Apply a reversible signed logarithmic transformation.

    The forward transform is ``sign(x) * log1p(abs(x))`` and the inverse is
    ``sign(x) * expm1(abs(x))``. Unlike ``log1p`` alone, this transformation
    supports both positive and negative finite values.
    """

    def fit(self, X, y=None):
        """Validate the input and remember the number of input features."""
        X = validate_data(
            self, X, reset=True, ensure_2d=True, dtype="numeric"
        )
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Transform values while preserving the input shape."""
        check_is_fitted(self, "n_features_in_")
        X = validate_data(
            self, X, reset=False, ensure_2d=True, dtype="numeric"
        )
        return np.sign(X) * np.log1p(np.abs(X))

    def inverse_transform(self, X):
        """Invert the signed logarithmic transformation."""
        check_is_fitted(self, "n_features_in_")
        X = validate_data(
            self, X, reset=False, ensure_2d=True, dtype="numeric"
        )
        return np.sign(X) * np.expm1(np.abs(X))