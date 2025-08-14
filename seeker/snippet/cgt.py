#date: 2025-08-14T16:49:36Z
#url: https://api.github.com/gists/7496556c009167934b0c0b92596eeec2
#owner: https://api.github.com/users/vahid-ahmadi

from policyengine_uk import Scenario, Microsimulation

DATASET = "hf://policyengine/policyengine-uk-data/enhanced_frs_2022_23.h5"

# Lower rate scenarios (baseline: 18%)
scenario_1 = Scenario(parameter_changes={
    "gov.hmrc.cgt.basic_rate": 0.19
})
scenario_2 = Scenario(parameter_changes={
    "gov.hmrc.cgt.basic_rate": 0.23
})
scenario_3 = Scenario(parameter_changes={
    "gov.hmrc.cgt.basic_rate": 0.28
})

# Higher rate scenarios (baseline: 24%)
scenario_4 = Scenario(parameter_changes={
    "gov.hmrc.cgt.higher_rate": 0.25,
    "gov.hmrc.cgt.additional_rate": 0.25
})
scenario_5 = Scenario(parameter_changes={
    "gov.hmrc.cgt.higher_rate": 0.29,
    "gov.hmrc.cgt.additional_rate": 0.29
})
scenario_6 = Scenario(parameter_changes={
    "gov.hmrc.cgt.higher_rate": 0.34,
    "gov.hmrc.cgt.additional_rate": 0.34
})

# Create baseline microsimulation
baseline = Microsimulation(dataset=DATASET)
baseline_income = baseline.calculate("household_net_income", 2026)

# Test all scenarios and report revenue impact
for name, scenario in [
    ("Basic rate +1 ppt (19%)", scenario_1), 
    ("Basic rate +5 ppt (23%)", scenario_2), 
    ("Basic rate +10 ppt (28%)", scenario_3),
    ("Higher/additional rate +1 ppt (25%)", scenario_4), 
    ("Higher/additional rate +5 ppt (29%)", scenario_5), 
    ("Higher/additional rate +10 ppt (34%)", scenario_6)
]:
    reformed = Microsimulation(dataset=DATASET, scenario=scenario)
    reformed_income = reformed.calculate("household_net_income", 2026)
    revenue = -(reformed_income - baseline_income).sum() / 1e9
    print(f"{name}: £{revenue:.1f}bn additional revenue per year")