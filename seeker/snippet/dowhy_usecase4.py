#date: 2022-05-30T17:03:21Z
#url: https://api.github.com/gists/704ad774d70ffdad46cb6a54dcadc509
#owner: https://api.github.com/users/misclassified

# Estimate
identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)

# Estimate Causal Effect with propensity score stratifications
estimate = model.estimate_effect(identified_estimand, 
                                 method_name="backdoor.propensity_score_stratification",
                                target_units="att")
print(f"Estimated average treatment effect on the treated {estimate.value}")