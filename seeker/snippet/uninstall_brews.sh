#date: 2022-01-06T17:19:49Z
#url: https://api.github.com/gists/79f1bfdcffab9006b64602a7d378fd67
#owner: https://api.github.com/users/pmarreck

while [[ `brew list | wc -l` -ne 0 ]]; do
    #Iterate over each installed package
    for EACH in `brew list`; do
        #Uninstall each package
        brew uninstall --ignore-dependencies $EACH --force
    done
done
