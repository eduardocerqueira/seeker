#date: 2023-10-03T17:03:41Z
#url: https://api.github.com/gists/7992f17d6a32afd9dabbd3526f305bf4
#owner: https://api.github.com/users/ojacques

aws sagemaker list-apps --query "Apps[?AppType=='KernelGateway' && Status=='InService']" | jq -r '.[] | "\(.AppName) \(.DomainId) \(.UserProfileName)"' | while read AppName DomainId UserProfileName; do echo Deleting $AppName for user $UserProfileName ... && aws sagemaker delete-app --domain-id $DomainId --app-type KernelGateway --app-name $AppName --user-profile $UserProfileName; done