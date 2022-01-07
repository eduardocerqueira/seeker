#date: 2022-01-07T17:08:15Z
#url: https://api.github.com/gists/d4bd6ae0006b5b6ed69f11365bbbba6c
#owner: https://api.github.com/users/lamesjaidler

# Access the rule_strings attribute from the generator step in the optimised pipeline
rule_strings = bs.pipeline_.get_params()['generator__rule_strings']
# Show the string representation of the rules
rule_strings