#date: 2023-02-16T17:08:59Z
#url: https://api.github.com/gists/2be1d07005a5d3071e9768d4fb3863c3
#owner: https://api.github.com/users/alexeldeib

# derived from custom cloud env?
REPO_DEPOT_ENDPOINT="{{AKSCustomCloudRepoDepotEndpoint}}"
# typically azureuser but can be user input
ADMINUSER={{GetParameter "linuxAdminUsername"}}
# unnecessary
MOBY_VERSION={{GetParameter "mobyVersion"}}
# environment derived, unnecessary?
TENANT_ID={{GetVariable "tenantID"}}
# cluster/node pool specific, derived from user input
KUBERNETES_VERSION={{GetParameter "kubernetesVersion"}}
# should be unnecessary
HYPERKUBE_URL={{GetParameter "kubernetesHyperkubeSpec"}}
# necessary only for non-cached versions
KUBE_BINARY_URL={{GetParameter "kubeBinaryURL"}}
# unnecessary
CUSTOM_KUBE_BINARY_URL={{GetParameter "customKubeBinaryURL"}}
# should be unneessary or bug
KUBEPROXY_URL={{GetParameter "kubeProxySpec"}}
# unique per cluster, actually not sure best way to extract?
APISERVER_PUBLIC_KEY={{GetParameter "apiServerCertificate"}}
# can be derived from environment/imds
SUBSCRIPTION_ID={{GetVariable "subscriptionId"}}
# can be derived from environment/imds
RESOURCE_GROUP={{GetVariable "resourceGroup"}}
# can be derived from environment/imds
LOCATION={{GetVariable "location"}}
# derived from cluster but unnecessary (?) only used by CCM
VM_TYPE={{GetVariable "vmType"}}
# derived from cluster but unnecessary (?) only used by CCM
SUBNET={{GetVariable "subnetName"}}
# derived from cluster but unnecessary (?) only used by CCM
NETWORK_SECURITY_GROUP={{GetVariable "nsgName"}}
# derived from cluster but unnecessary (?) only used by CCM
VIRTUAL_NETWORK={{GetVariable "virtualNetworkName"}}
# derived from cluster but unnecessary (?) only used by CCM
VIRTUAL_NETWORK_RESOURCE_GROUP={{GetVariable "virtualNetworkResourceGroupName"}}
# derived from cluster but unnecessary (?) only used by CCM
ROUTE_TABLE={{GetVariable "routeTableName"}}
# derived from cluster but unnecessary (?) only used by CCM
PRIMARY_AVAILABILITY_SET={{GetVariable "primaryAvailabilitySetName"}}
# derived from cluster but unnecessary (?) only used by CCM
PRIMARY_SCALE_SET={{GetVariable "primaryScaleSetName"}}
# user input
SERVICE_PRINCIPAL_CLIENT_ID={{GetParameter "servicePrincipalClientId"}}
# user input
NETWORK_PLUGIN={{GetParameter "networkPlugin"}}
# user input
NETWORK_POLICY={{GetParameter "networkPolicy"}}
# unnecessary
VNET_CNI_PLUGINS_URL={{GetParameter "vnetCniLinuxPluginsURL"}}
# unnecessary
CNI_PLUGINS_URL={{GetParameter "cniPluginsURL"}}
# kubelet config for azure stuff, static/derived from user inputs.
### BEGIN CLOUD CONFIG
CLOUDPROVIDER_BACKOFF={{GetParameterProperty "cloudproviderConfig" "cloudProviderBackoff"}}
CLOUDPROVIDER_BACKOFF_MODE={{GetParameterProperty "cloudproviderConfig" "cloudProviderBackoffMode"}}
CLOUDPROVIDER_BACKOFF_RETRIES={{GetParameterProperty "cloudproviderConfig" "cloudProviderBackoffRetries"}}
CLOUDPROVIDER_BACKOFF_EXPONENT={{GetParameterProperty "cloudproviderConfig" "cloudProviderBackoffExponent"}}
CLOUDPROVIDER_BACKOFF_DURATION={{GetParameterProperty "cloudproviderConfig" "cloudProviderBackoffDuration"}}
CLOUDPROVIDER_BACKOFF_JITTER={{GetParameterProperty "cloudproviderConfig" "cloudProviderBackoffJitter"}}
CLOUDPROVIDER_RATELIMIT={{GetParameterProperty "cloudproviderConfig" "cloudProviderRateLimit"}}
CLOUDPROVIDER_RATELIMIT_QPS={{GetParameterProperty "cloudproviderConfig" "cloudProviderRateLimitQPS"}}
CLOUDPROVIDER_RATELIMIT_QPS_WRITE={{GetParameterProperty "cloudproviderConfig" "cloudProviderRateLimitQPSWrite"}}
CLOUDPROVIDER_RATELIMIT_BUCKET={{GetParameterProperty "cloudproviderConfig" "cloudProviderRateLimitBucket"}}
CLOUDPROVIDER_RATELIMIT_BUCKET_WRITE={{GetParameterProperty "cloudproviderConfig" "cloudProviderRateLimitBucketWrite"}}
LOAD_BALANCER_DISABLE_OUTBOUND_SNAT={{GetParameterProperty "cloudproviderConfig" "cloudProviderDisableOutboundSNAT"}}
USE_MANAGED_IDENTITY_EXTENSION={{GetVariable "useManagedIdentityExtension"}}
USE_INSTANCE_METADATA={{GetVariable "useInstanceMetadata"}}
LOAD_BALANCER_SKU={{GetVariable "loadBalancerSku"}}
EXCLUDE_MASTER_FROM_STANDARD_LB={{GetVariable "excludeMasterFromStandardLB"}}
MAXIMUM_LOADBALANCER_RULE_COUNT={{GetVariable "maximumLoadBalancerRuleCount"}}
### END CLOUD CONFIG
# always containerd.
CONTAINER_RUNTIME={{GetParameter "containerRuntime"}}
# static/unnecesary
CLI_TOOL={{GetParameter "cliTool"}}
# unecessary
CONTAINERD_DOWNLOAD_URL_BASE={{GetParameter "containerdDownloadURLBase"}}
# user input
NETWORK_MODE={{GetParameter "networkMode"}}
# static-ish
KUBE_BINARY_URL={{GetParameter "kubeBinaryURL"}}
# user input.
USER_ASSIGNED_IDENTITY_ID={{GetVariable "userAssignedIdentityID"}}
# unique per cluster
API_SERVER_NAME={{GetKubernetesEndpoint}}
# static-ish
IS_VHD={{GetVariable "isVHD"}}
# derived from VM size
GPU_NODE={{GetVariable "gpuNode"}}
# unused
SGX_NODE={{GetVariable "sgxNode"}}
# user input
MIG_NODE={{GetVariable "migNode"}}
# depends on hardware, unnecessary for oss, but aks provisions gpu drivers.
CONFIG_GPU_DRIVER_IF_NEEDED={{GetVariable "configGPUDriverIfNeeded"}}
# deprecated/preview only, don't do this for OSS
ENABLE_GPU_DEVICE_PLUGIN_IF_NEEDED={{GetVariable "enableGPUDevicePluginIfNeeded"}}
# user input, don't do this for OSS
TELEPORTD_PLUGIN_DOWNLOAD_URL={{GetParameter "teleportdPluginURL"}}
# unused
CONTAINERD_VERSION={{GetParameter "containerdVersion"}}
# only for testing
CONTAINERD_PACKAGE_URL={{GetParameter "containerdPackageURL"}}
# unused
RUNC_VERSION={{GetParameter "runcVersion"}}
# testing only
RUNC_PACKAGE_URL={{GetParameter "runcPackageURL"}}
# derived from private cluster user input...I think?
ENABLE_HOSTS_CONFIG_AGENT="{{EnableHostsConfigAgent}}"
# user input
DISABLE_SSH="{{ShouldDisableSSH}}"
# static true
NEEDS_CONTAINERD="{{NeedsContainerd}}"
# user input
TELEPORT_ENABLED="{{TeleportEnabled}}"
# user input
SHOULD_CONFIGURE_HTTP_PROXY="{{ShouldConfigureHTTPProxy}}"
# user input
SHOULD_CONFIGURE_HTTP_PROXY_CA="{{ShouldConfigureHTTPProxyCA}}"
# user input
HTTP_PROXY_TRUSTED_CA="{{GetHTTPProxyCA}}"
# user input
SHOULD_CONFIGURE_CUSTOM_CA_TRUST="{{ShouldConfigureCustomCATrust}}"
# user input
CUSTOM_CA_TRUST_COUNT="{{len GetCustomCATrustConfigCerts}}"
# user input
{{range $i, $cert := GetCustomCATrustConfigCerts}}
CUSTOM_CA_CERT_{{$i}}="{{$cert}}"
{{end}}
# user input
IS_KRUSTLET="{{IsKrustlet}}"
# determined by GPU hardware type
GPU_NEEDS_FABRIC_MANAGER="{{GPUNeedsFabricManager}}"
# user input
NEEDS_DOCKER_LOGIN="{{and IsDockerContainerRuntime HasPrivateAzureRegistryServer}}"
# user input
IPV6_DUAL_STACK_ENABLED="{{IsIPv6DualStackFeatureEnabled}}"
# mostly static/can be
OUTBOUND_COMMAND="{{GetOutboundCommand}}"
# user input
ENABLE_UNATTENDED_UPGRADES="{{EnableUnattendedUpgrade}}"
# derived
ENSURE_NO_DUPE_PROMISCUOUS_BRIDGE="{{ and NeedsContainerd IsKubenet (not HasCalicoNetworkPolicy) }}"
# user input
SHOULD_CONFIG_SWAP_FILE="{{ShouldConfigSwapFile}}"
# user input
SHOULD_CONFIG_TRANSPARENT_HUGE_PAGE="{{ShouldConfigTransparentHugePage}}"
{{/* both CLOUD and ENVIRONMENT have special values when IsAKSCustomCloud == true */}}
{{/* CLOUD uses AzureStackCloud and seems to be used by kubelet, k8s cloud provider */}}
{{/* target environment seems to go to ARM SDK config */}}
{{/* not sure why separate/inconsistent? */}}
{{/* see GetCustomEnvironmentJSON for more weirdness. */}}
# derive from environment/user input
TARGET_CLOUD="{{- if IsAKSCustomCloud -}} AzureStackCloud {{- else -}} {{GetTargetEnvironment}} {{- end -}}"
# derive from environment/user input
TARGET_ENVIRONMENT="{{GetTargetEnvironment}}"
# derive from environment/user input
CUSTOM_ENV_JSON="{{GetBase64EncodedEnvironmentJSON}}"
# derive from environment/user input
IS_CUSTOM_CLOUD="{{IsAKSCustomCloud}}"
# static
CSE_HELPERS_FILEPATH="{{GetCSEHelpersScriptFilepath}}"
# static
CSE_DISTRO_HELPERS_FILEPATH="{{GetCSEHelpersScriptDistroFilepath}}"
# static
CSE_INSTALL_FILEPATH="{{GetCSEInstallScriptFilepath}}"
# static
CSE_DISTRO_INSTALL_FILEPATH="{{GetCSEInstallScriptDistroFilepath}}"
# static
CSE_CONFIG_FILEPATH="{{GetCSEConfigScriptFilepath}}"
# user input
AZURE_PRIVATE_REGISTRY_SERVER="{{GetPrivateAzureRegistryServer}}"
# user input
HAS_CUSTOM_SEARCH_DOMAIN="{{HasCustomSearchDomain}}"
# static
CUSTOM_SEARCH_DOMAIN_FILEPATH="{{GetCustomSearchDomainsCSEScriptFilepath}}"
# user input
HTTP_PROXY_URLS="{{GetHTTPProxy}}"
# user input
HTTPS_PROXY_URLS="{{GetHTTPSProxy}}"
# user input
NO_PROXY_URLS="{{GetNoProxy}}"
# static true
CLIENT_TLS_BOOTSTRAPPING_ENABLED="{{IsKubeletClientTLSBootstrappingEnabled}}"
# derived from user input
DHCPV6_SERVICE_FILEPATH="{{GetDHCPv6ServiceCSEScriptFilepath}}"
# derived from user input
DHCPV6_CONFIG_FILEPATH="{{GetDHCPv6ConfigCSEScriptFilepath}}"
# user input
THP_ENABLED="{{GetTransparentHugePageEnabled}}"
# user input
THP_DEFRAG="{{GetTransparentHugePageDefrag}}"
# only required for RP cluster.
SERVICE_PRINCIPAL_FILE_CONTENT= "**********"
# unnecessary
KUBELET_CLIENT_CONTENT="{{GetKubeletClientKey}}"
# unnecessary
KUBELET_CLIENT_CERT_CONTENT="{{GetKubeletClientCert}}"
# can be static.
KUBELET_CONFIG_FILE_ENABLED="{{IsKubeletConfigFileEnabled}}"
# mix of user/static/RP-generated.
KUBELET_CONFIG_FILE_CONTENT="{{GetKubeletConfigFileContentBase64}}"
# user input.
SWAP_FILE_SIZE_MB="{{GetSwapFileSizeMB}}"
# determine by OS + GPU hardware requirements
# can be determined automatically, but hard.
# suggest using GPU operator.
GPU_DRIVER_VERSION="{{GPUDriverVersion}}"
# user-specified
GPU_INSTANCE_PROFILE="{{GetGPUInstanceProfile}}"
# user-specified
CUSTOM_SEARCH_DOMAIN_NAME="{{GetSearchDomainName}}"
# user-specified
CUSTOM_SEARCH_REALM_USER="{{GetSearchDomainRealmUser}}"
# user-specified
CUSTOM_SEARCH_REALM_PASSWORD= "**********"
# user-specified
MESSAGE_OF_THE_DAY="{{GetMessageOfTheDay}}"
# user-specified
HAS_KUBELET_DISK_TYPE="{{HasKubeletDiskType}}"
# can be automatically determined.
NEEDS_CGROUPV2="{{Is2204VHD}}"
# user-specified.
SYSCTL_CONTENT="{{GetSysctlContent}}"
# nodepool or node specific. can be created automatically.
TLS_BOOTSTRAP_TOKEN= "**********"
# unique per nodepool. partially user-specified, static, and RP-generated.
KUBELET_FLAGS="{{GetKubeletConfigKeyVals}}"
# unique per cluster. user-specified.
NETWORK_POLICY="{{GetParameter "networkPolicy"}}"
# node-pool specific. user-specified.
KUBELET_NODE_LABELS="{{GetAgentKubernetesLabels . }}"
# can be made static.
AZURE_ENVIRONMENT_FILEPATH="{{- if IsAKSCustomCloud}}/etc/kubernetes/{{GetTargetEnvironment}}.json{{end}}"
# unique per cluster.
KUBE_CA_CRT="{{GetParameter "caCertificate"}}"
# Static
KUBENET_TEMPLATE="{{GetKubenetTemplate}}"
# determined by GPU VM size, WASM support, Kata support.
CONTAINERD_CONFIG_CONTENT="{{GetContainerdConfigContent}}" 
ort.
CONTAINERD_CONFIG_CONTENT="{{GetContainerdConfigContent}}" 
