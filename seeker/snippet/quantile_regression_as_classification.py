#date: 2023-11-29T16:43:29Z
#url: https://api.github.com/gists/43158d8a3654f5ed051b20840308939e
#owner: https://api.github.com/users/ogrisel

# %%
from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.validation import check_consistent_length
from sklearn.utils import check_random_state
import numpy as np


class BinnedQuantileRegressor(BaseEstimator, RegressorMixin):
    def __init__(
            self,
            estimator=None,
            n_bins=100,
            quantile=0.5,
            binning_strategy="uniform",
            interpolation_knots="midpoints",
            interpolation_kind="linear",
            random_state=None,
        ):
        self.n_bins = n_bins
        self.estimator = estimator
        self.quantile = quantile
        self.binning_strategy = binning_strategy
        self.interpolation_knots = interpolation_knots
        self.interpolation_kind = interpolation_kind
        self.random_state = random_state

    def fit(self, X, y):
        assert self.interpolation_knots in ("edges", "midpoints")
        # Lightweight input validation: most of the input validation will be
        # handled by the sub estimators.
        random_state = check_random_state(self.random_state)
        check_consistent_length(X, y)
        self.target_binner_ = KBinsDiscretizer(
            n_bins=self.n_bins,
            strategy=self.binning_strategy,
            subsample=200_000,
            encode="ordinal",
            random_state=random_state,
        )

        y_binned = self.target_binner_.fit_transform(
            np.asarray(y).reshape(-1, 1)
        ).ravel().astype(np.int32)

        # Fit the multiclass classifier to predict the binned targets from the
        # training set.
        if self.estimator is None:
            estimator = RandomForestClassifier(random_state=random_state)
        else:
            estimator = clone(self.estimator)
        self.estimator_ = estimator.fit(X, y_binned)
        return self

    def predict_quantiles(self, X, quantiles):
        check_is_fitted(self, "estimator_")
        edges = self.target_binner_.bin_edges_[0]
        n_bins = edges.shape[0] - 1
        expected_shape = (X.shape[0], n_bins)
        y_proba_raw = self.estimator_.predict_proba(X)
        
        # Some bins might stay empty on the training set, in particular with the
        # uniform binning strategy. Typically, classifiers do not learn to
        # predict an explicit 0 probability for unobserved classes so we have
        # to post process their output:
        if y_proba_raw.shape != expected_shape:
            y_proba = np.zeros(shape=expected_shape)
            y_proba[:, self.estimator_.classes_] = y_proba_raw
        else:
            y_proba = y_proba_raw

        # Build the mapper for inverse CDF mapping, from cumulated
        # probabilities to continuous prediction.
        if self.interpolation_knots == "edges":
            y_cdf = np.zeros(shape=(X.shape[0], edges.shape[0]))
            y_cdf[:, 1:] = np.cumsum(y_proba, axis=1)
            return np.asarray(
                [
                    interp1d(
                        y_cdf_i,
                        edges,
                        kind=self.interpolation_kind,
                    )(quantiles)
                    for y_cdf_i in y_cdf
                ]
            )
        else:
            midpoints = (edges[1:] + edges[:-1]) / 2
            y_cdf = np.cumsum(y_proba, axis=1)
            return np.asarray(
                [
                    interp1d(
                        y_cdf_i,
                        midpoints,
                        kind=self.interpolation_kind,
                        bounds_error=False,
                        fill_value=(midpoints[0], midpoints[-1]),
                    )(quantiles)
                    for y_cdf_i in y_cdf
                ]
            )

    def predict(self, X):
        return self.predict_quantiles(X, self.quantile).ravel()


# %%
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer
from sklearn.metrics import mean_pinball_loss
from functools import partial

rng = np.random.RandomState(42)
x = rng.uniform(-1, 1, size=10_000)
X = x.reshape(-1, 1)
y = np.sin(x * np.pi) + 1
y /= y.max()

# Add heteroscedastic noise to make conditional quantiles interesting:
y += rng.normal(0, y / 5, size=y.shape)
y
plt.plot(x, y, "o", alpha=0.1)

base_classifier = RandomForestClassifier(random_state=42, min_samples_leaf=50)

# Using the XGBoost classifier as a base estimator gives similar results but it typically
# slower and not better than RF with tuned min_samples_leaf.

# base_classifier = XGBClassifier(
#     tree_method="hist",
#     multi_strategy="multi_output_tree",
#     # nthread=1,  # set to serial to make it easier to understand the timings
#     n_estimators=100,
# )

# XXX: the following does not work at all! something must be wrong but I could
# not spot it by tweaking the parameters.

# base_classifier = HistGradientBoostingClassifier(
#     random_state=42,
#     early_stopping=True,
#     max_iter=1_000,
#     min_samples_leaf=1,
# )

binned_quantile_reg = BinnedQuantileRegressor(
    estimator=base_classifier,
    n_bins=100,
    interpolation_knots="midpoints",
    binning_strategy="quantile",
    quantile=0.95,
)
X_test = np.linspace(-1, 1, 1000).reshape(-1, 1)

q_pred = binned_quantile_reg.fit(X, y).predict_quantiles(X_test, [0.05, 0.95])
q05_pred, q95_pred = q_pred.T
plt.plot(X_test, q05_pred, label="q05")
plt.plot(X_test, q95_pred, label="q95")
# Plot horizontal lines for each of the bin edges:
thresholds = binned_quantile_reg.target_binner_.bin_edges_[0]
plt.hlines(thresholds, X_test.min(), X_test.max(), color="k", linestyles="--", alpha=0.2)
plt.legend()
plt.show()


def evaluate(model, X, y):
    print(model)
    neg_q95_pinball_loss = make_scorer(
        partial(mean_pinball_loss, alpha=0.95),
        greater_is_better=False,
    )
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=5,
        scoring=neg_q95_pinball_loss,
        return_train_score=True,
        n_jobs=-1,
    )
    train_scores = -cv_results["train_score"]
    val_scores = -cv_results["test_score"]
    fit_time = cv_results["fit_time"]
    print(
        f"Train q0.95 pinball: {train_scores.mean():.4f} +/- {train_scores.std():.4f}\n"
        f"Val q0.95 pinball: {val_scores.mean():.4f} +/- {val_scores.std():.4f}\n"
        f"Fit time: {fit_time.mean():.4f} +/- {fit_time.std():.4f} s"
    )
evaluate(binned_quantile_reg, X, y)

# %%
from sklearn.ensemble import HistGradientBoostingRegressor

evaluate(
    HistGradientBoostingRegressor(
        random_state=0,
        loss="quantile",
        quantile=0.95,
        early_stopping=True,
        n_iter_no_change=2,
        max_iter=1_000,
    ),
    X,
    y,
)

# %%
from xgboost import XGBRegressor

evaluate(
    XGBRegressor(
        random_state=42,
        tree_method="hist",
        objective="reg:quantileerror",
        quantile_alpha=0.95,
    ),
    X,
    y,
)
