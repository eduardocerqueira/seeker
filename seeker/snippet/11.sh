#date: 2022-10-04T17:04:34Z
#url: https://api.github.com/gists/68f1152c50f88752493a2b5ae944caef
#owner: https://api.github.com/users/jeromedecoster

# used to pull image from private ECR by the deployment manifest
kubectl create secret docker-registry regcred -n my-app \
    --docker-server=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com \
    --docker-username=AWS \
    --docker-password= "**********"-login-password --region $AWS_REGION)