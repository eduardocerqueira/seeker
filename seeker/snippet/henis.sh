#date: 2021-11-29T17:13:20Z
#url: https://api.github.com/gists/8576af42a63e42472b3242cd7cef0e9a
#owner: https://api.github.com/users/ivica-k

$ python henis.py
Total number of ENIs: 409
Total number of hyperplane ENIs: 139
Total number of hyperplane ENIs used by Lambdas: 68

# adding --verbose shows ENI and HENI IDs
$ python henis.py --verbose
Total number of ENIs: 409
Total number of hyperplane ENIs: 139
The list of hyperplane ENIs: ['eni-04910655b12d5071d' ...]
# ... SNIP ...
Total number of hyperplane ENIs used by Lambdas: 68
The list of hyperplane ENIs associated with Lambdas: ['eni-064cf287ea3ba1a8c' ...]