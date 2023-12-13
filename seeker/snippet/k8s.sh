#date: 2023-12-13T17:09:16Z
#url: https://api.github.com/gists/5e28b1f4d2420c403f66cd86c1476d54
#owner: https://api.github.com/users/lgualpa81

Issue:
Error: Kubernetes cluster unreachable: exec plugin: invalid apiVersion "client.authentication.k8s.io/v1alpha1"

provider "helm" {
  kubernetes {
    config_path            = "~/.kube/config"
    host                   = module.eks.cluster_endpoint
    token                  = "**********"
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    exec {
      api_version = "client.authentication.k8s.io/v1alpha1"
      command     = "aws"
      args        = "**********"
    }
  }
}

Check EKS cluster apiVersion with AWS CLI

$ aws eks get-token --cluster-name YOUR_CLUSTER_NAME --profile YOUR_AWS_PROFILE | jq .apiVersion

Result: "client.authentication.k8s.io/v1beta1"

Solution:
Update apiVersion "v1alpha1" to "v1beta1"

Reference:
https://github.com/hashicorp/terraform-provider-helm/issues/893eta1"

Reference:
https://github.com/hashicorp/terraform-provider-helm/issues/893