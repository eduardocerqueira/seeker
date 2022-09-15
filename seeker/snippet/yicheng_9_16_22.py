#date: 2022-09-15T17:19:55Z
#url: https://api.github.com/gists/ab67f185f13294b1642fa19a671155af
#owner: https://api.github.com/users/jimmyguerrero

# Only omits None-valued samples
view = dataset.exists("ground_truth")

# Also omits samples with empty Detections([])
view = dataset.match(F("ground_truth.detections").length() > 0)