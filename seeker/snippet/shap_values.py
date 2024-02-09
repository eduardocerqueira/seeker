#date: 2024-02-09T17:03:32Z
#url: https://api.github.com/gists/696301eeb9ae8b2686fc7c1d2aaa6a6c
#owner: https://api.github.com/users/agatheminaro

import shap

# Assuming the model is named `lgbm`
explainer = shap.TreeExplainer(lgbm)
shap_values = explainer.shap_values(X_train)

shap.summary_plot(shap_values, X_train, plot_type="bar")
