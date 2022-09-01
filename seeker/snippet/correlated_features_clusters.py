#date: 2022-09-01T17:10:54Z
#url: https://api.github.com/gists/c979dfaaeace1614389f8c64ba10bd88
#owner: https://api.github.com/users/MatheusHam

from feature_engine.selection import SmartCorrelatedSelection


MODEL_TYPE = "classifier" ## Or "regressor"
CORRELATION_THRESHOLD = .97

# Setup Smart Selector /// Tks feature_engine
feature_selector = SmartCorrelatedSelection(
    variables=None,
    method="spearman",
    threshold=CORRELATION_THRESHOLD,
    missing_values="ignore",
    selection_method="variance",
    estimator=None,
)


feature_selector.fit_transform(X)

### Setup a list of correlated clusters as lists and a list of uncorrelated features
correlated_sets = feature_selector.correlated_feature_sets_

correlated_clusters = [list(feature) for feature in correlated_sets]

correlated_features = [feature for features in correlated_clusters for feature in features]

uncorrelated_features = [feature for feature in X if feature not in correlated_features]