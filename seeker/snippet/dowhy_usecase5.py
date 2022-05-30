#date: 2022-05-30T17:07:00Z
#url: https://api.github.com/gists/2db5805d8bb7e10b3b77f528943fcac4
#owner: https://api.github.com/users/misclassified

# Refutation test
refutation = model.refute_estimate(identified_estimand, 
                                   estimate,
                                   method_name="placebo_treatment_refuter",
                                   placebo_type="permute", num_simulations=20)