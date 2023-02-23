#date: 2023-02-23T17:02:11Z
#url: https://api.github.com/gists/cd70b5203206e8b7a2f526dacab4454e
#owner: https://api.github.com/users/joshreini1

# import trushap, and optionally alias as shap to preserve your SHAP code.
from truera.client.experimental.trushap import trushap as shap

# Initialize the explainer. 
# Include TruEra authentication (optional) to add the model to your TruEra deployment.
explainer = "**********"= CONNECTION_STRING, token = TOKEN)

# Calculate shapley values AND add data split to your TruEra deployment.
shap_values = explainer(X).
shap_values = explainer(X)