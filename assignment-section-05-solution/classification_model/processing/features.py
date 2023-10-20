from sklearn.base import BaseEstimator, TransformerMixin


class ExtractLetterTransformer(BaseEstimator, TransformerMixin):
    # Extract the first letter

    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        # Not needed
        return self

    def transform(self, X):
        for v in self.variables:
            X[v] = X[v].str[0]
        return X
