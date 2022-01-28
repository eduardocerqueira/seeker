#date: 2022-01-28T17:14:32Z
#url: https://api.github.com/gists/95ea3815d7480f1c15dd4bc7871c56f4
#owner: https://api.github.com/users/VladStarr

kubectl get deploy --output=jsonpath='
{range .items[*]}
{.metadata.name}{":"}
{"\n\n"}
{.spec.template.spec.containers[0].args[*]}
{end}' | perl -pe 's/ /\n/g'