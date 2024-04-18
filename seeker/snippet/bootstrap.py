#date: 2024-04-18T16:51:32Z
#url: https://api.github.com/gists/9a41b250a6504884e6797174749f6a94
#owner: https://api.github.com/users/atistler

 <#
    .SYNOPSIS
      Registers this host with ssm agent

    .DESCRIPTION
      Downloads the agent, installs it, must be run as Administrator

    .PARAMETER Platform
      Device platform such as vmware, dedicated (CORE physical), etc.
      It's used by the script to distinguish where to fetch the JWT token from.

    .PARAMETER HttpProxy
      Proxy host to be used by the bootstrap script when making external calls.
      It will also be set in the SSM agent's config after installation in order
      to be used during the registering phase.

      The input should be in the "<http|https>://<fqdn|ip>:<port>" format. For example: "http://10.22.33.44:4242"

    .PARAMETER InstallerDownloadRegion
      The AWS region to download the agent from. If not provided, the default
      agent package URL will be used. For example: "eu-central-1"

    .NOTES
      Version:          2.0
      Author:           adam.tistler@rackspace.com
      Creation Date:    09/21/22
#>
<#
    .Description
     Exit Codes : description
    100: Failed to install Package 'AmazonSSMAgent
    101: Failed to uninstall Package 'AmazonSSMAg
    103: Unable to stop 'AmazonSsmAgent' service
    104: Service 'AmazonSsmAgent' has not reached a 'running' status after multiple attempts.
    105: GET request to activation job url failed.
    106: SSM agent activation job was not successful
    107: Agent activation job did not complete after multiple attempts
    108: POST request to activation url failed.
    109: POST request to activation url did not return a Location header.
    110: Failed to execute agent registration command.
    111: Failed to execute agent diagnostics command.
    112: Management agent was not registered successfully.
    113: ssm-cli.exe' command does not exist
    114: Failed to execute clear agent registration command
    115: HTTP request failed while installing Amazon SSM Agent
    116: Package 'AmazonSsmAgent' is not currently installed
    117: HTTP request failed while uninstalling Amazon SSM Agent
    118: "**********"
    119: "**********"
    120: GET request to metadata instance url failed.
    121: GET request to metadata attest url failed.
    122: "**********"
    123: "**********"
    124: GET request to instance identity url failed
    125: GET request to instance metadata identity url failed
    126: Downloading agent package installer from package installer url failed
    127: Command 'amazon-ssm-agent' not found
    128: SSM package is not installed, cannot reregister agent
    129: ssm-cli' command does not exist
    133: The Rackspace management agent only supports Windows Server or Windows Domain Controllers, it does not support this product type.
    134: The Rackspace management agent supports Windows versions ['2012', '2016', '2019', '2022'], it does not support this version.
    135: The Rackspace management agent only supports 64-bit versions of Windows.
    136: rpctool' command does not exist
    144: Something went wrong. Uncaught exception occurred.
#>
param(
    [Parameter(Mandatory = $true)][ValidateSet("Install", "Reregister", "Uninstall")]
    [string]$Command,

    [Parameter(Mandatory = $true)][ValidateSet("Dedicated", "VMWare", "GCP", "Azure")]
    [string]$Platform,

    [Parameter()][ValidatePattern('^(http|https):\/\/([-a-zA-Z0-9._~!$&\"*+,;=:(PCTENCODED)]*\.[a-z]{2,}|[-a-zA-Z0-9._~!$&\"*+,;=:(PCTENCODED)]*|[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}):\d{2,5}')]
    [string]$HttpProxy, # valid "<http|https>://<fqdn|ip>:<port>" format

    [Parameter(Mandatory = $false)][ValidatePattern('([a-z]{2})\-([a-z]+)-(\d+)')]
    [string]$InstallerDownloadRegion,

    [Parameter(Mandatory = $false)][ValidateSet("text", "json")]
    [string]$ResultFormat = "text"
)

$SupportedWindowsVersions = @("2012", "2016", "2019", "2022")
$PlatformServicesBaseUrl = "https://add-ons.api.manage.rackspace.com/v1.0"
$DefaultAgentPackageUrl = "https://s3.amazonaws.com/ec2-downloads-windows/SSMAgent/latest/windows_amd64/AmazonSSMAgentSetup.exe"
$RegionalAgentPackageUrl = "https://amazon-ssm-{0}.s3.{0}.amazonaws.com/latest/windows_amd64/AmazonSSMAgentSetup.exe"
$IgnoreTagKey = "rackspace-addon-ignore"
$SsmService = "AmazonSsmAgent"
$SsmProcess = "amazon-ssm-agent"
$AgentPackageName = "Amazon SSM Agent"
$RegistrationWait = 5
$LogPath = [Environment]::CurrentDirectory
$LogName = "agent_bootstrap.log"
$AgentProgramDir = "C:\Program Files\Amazon\SSM"
$AgentDataDir = "$Env:ProgramData\Amazon\SSM"
$DebugPreference = "Continue"
$ErrorActionPreference = "Stop"
$ResultDelimiterFormat = "$('-'*25){0}$('-'*25)"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Add-Type -AssemblyName System.Net.Http

function WriteLog {
    param(
        [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$Message,
        [Parameter()][ValidateNotNullOrEmpty()][ValidateSet('DEBUG', 'WARN')][string]$Level = 'DEBUG'
    )
    $FullPath = Join-Path -Path $LogPath -ChildPath $LogName
    If (!(Test-Path -Path $LogPath)) {
        New-Item -Path $LogPath -ItemType Directory
    }
    If (!(Test-Path -Path $FullPath -PathType Leaf)) {
        New-Item -Path $FullPath -ItemType File
    }
    $Line = AsText -Message $Message -Level $Level
    Add-Content -Path $FullPath -Value $Line
    return $Line
}

function AsText {
    param(
        [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$Message,
        [Parameter()][ValidateNotNullOrEmpty()][ValidateSet('DEBUG', 'WARN')][string]$Level = 'DEBUG'
    )
    $Format = "{0} {1} - {2}"
    $Line = ($Format -f ((Get-Date).ToUniversalTime().ToString("o")), $Level, $Message)
    return $Line

}

function AsJSON {
    param(
        [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$Message,
        [Parameter()][ValidateNotNullOrEmpty()][ValidateSet('DEBUG', 'WARN')][string]$Level = 'DEBUG',
        [Parameter()][string]$Details = ''
    )
    return @{
        timestamp = ((Get-Date).ToUniversalTime().ToString("o"))
        level     = $Level
        message   = $Message
        details   = $Details
    } | ConvertTo-Json -Compress

}

function Die {
    param(
        [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$Message,
        [Parameter(Mandatory = $true)][int]$ExitCode,
        [Parameter(Mandatory = $false)][string]$Details = ''
    )
    if ($Details -ne '') {
        $DetailedMessage = "${Message} `r`n ${Details}"
    }
    else {
        $DetailedMessage = $Message
    }
    $Line = WriteLog -Level 'WARN' -Message $DetailedMessage # Ignoring output to prevent writing to console twice
    Write-Warning ($ResultDelimiterFormat -f "Failed")
    if ($ResultFormat -eq 'json') {
        Write-Warning ( AsJSON -Level 'WARN' -Message $Message -Details $Details )
    }
    else {
        Write-Warning ( AsText -Level 'WARN' -Message $DetailedMessage )
    }
    exit $ExitCode
}

function Success {
    param(
        [Parameter(Mandatory = $true)][ValidateNotNullOrEmpty()][string]$Message,
        [Parameter(Mandatory = $false)][string]$Details = ''
    )
    if ($Details -ne '') {
        $DetailedMessage = "${Message} `r`n ${Details}"
    }
    else {
        $DetailedMessage = $Message
    }
    $Line = WriteLog -Level 'DEBUG' -Message $DetailedMessage  # Ignoring output to prevent writing to console twice
    Write-Debug ($ResultDelimiterFormat -f "Success")
    if ($ResultFormat -eq 'json') {
        Write-Debug ( AsJSON -Level 'DEBUG' -Message $Message -Details $Details )
    }
    else {
        Write-Debug ( AsText -Level 'DEBUG' -Message $DetailedMessage )
    }
    exit 0
}

function IsTruthy([AllowNull()]$Value) {
    return (($Value -is [string]) -and ('true', 't', 'yes', 'y', '1').Contains($Value.toLower()))
}

function VmwarePlatform {
    function GetToken {
        $GetTokenAttempts = "**********"
        $GetTokenDelay = "**********"
        $RpcToolPath = "${Env:ProgramFiles}\VMware\VMware Tools\rpctool.exe"
        if (!(Test-Path -Path $RpcToolPath)) {
            Die "${RpcToolPath} command does not exist" -ExitCode 136
        }
        try {
            foreach ($Attempt in 1..$GetTokenAttempts) {
                WriteLog "Running vmware get token command (attempt ${Attempt}): "**********"
                $Token = "**********"
                if ($Token) {
                    break
                }
                Start-Sleep -Seconds $GetTokenDelay
            }
            if ($Token) {
                WriteLog "Found vmware auth token: "**********"
                return $Token
            }
            Die "Could not retrieve token after ${GetTokenAttempt} attempts.  Rackspace vmware provisioning is responsible for setting the 'guestinfo.machine.id' custom property." -ExitCode 118
        }
        catch {
            Die "Vmware get token command failed: "**********"
        }
    }
    function IsAgentDisabled {
        WriteLog "Platform 'vmware' does not support bypassing agent bootstrap using tags" | Write-Debug
        return $false
    }
    return @{
        ActivationUrl   = "${PlatformServicesBaseUrl}/instance/activate"
        Name            = "vmware"
         "**********"        = $Function: "**********"
        IsAgentDisabled = $Function:IsAgentDisabled
    }
}

function AzurePlatform {
    function GetToken {
        $InstanceMetadata = GetMetadata
        $AttestDoc = GetAttestDocument

        $TokenData = "**********"
            instance  = @{
                subscriptionId = $InstanceMetadata.subscriptionId
                location       = $InstanceMetadata.location
                name           = $InstanceMetadata.name
                vmId           = $InstanceMetadata.vmId
                vmScaleSetName = $InstanceMetadata.vmScaleSetName
            }
            signature = $AttestDoc.signature
            encoding  = $AttestDoc.encoding
        }
        return ( ConvertTo-Json -InputObject $TokenData -Compress )
    }
    function IsAgentDisabled {
        WriteLog "Checking Azure metadata for '${IgnoreTagKey}' tag" | Write-Debug
        $Tags = GetInstanceTags
        return IsTruthy($Tags[$IgnoreTagKey])
    }
    function GetInstanceTags {
        $Tags = @{ }
        $Metadata = GetMetadata
        ForEach-Object -InputObject $Metadata["tagsList"] {
            $Tags[$_.name] = $_.value
        }
        return $Tags
    }
    function GetMetadata {
        $InstanceMetadataUrl = "http://169.254.169.254/metadata/instance/compute?api-version=2021-01-01&format=json"
        WriteLog "Making GET request to azure instance metadata url: ${InstanceMetadataUrl}" | Write-Debug
        $Params = @{
            Uri             = $InstanceMetadataUrl
            Headers         = @{ Metadata = "true" }
            UseBasicParsing = $true
        }
        try {
            $Response = Invoke-WebRequest @Params
        }
        catch [System.Net.Http.HttpRequestException], [System.Net.WebException] {
            Die "GET request to metadata instance url ${InstanceMetadataUrl} failed: $( $_.Exception.Message )" -ExitCode 120
        }
        return ConvertFrom-Json $Response.Content
    }
    function GetAttestDocument {
        $AttestUrl = "http://169.254.169.254/metadata/attested/document?api-version=2021-01-01"
        WriteLog "Making GET request to azure instance attest document url: ${InstanceMetadataUrl}" | Write-Debug
        $Params = @{
            Uri             = $AttestUrl
            Headers         = @{ Metadata = "true" }
            UseBasicParsing = $true
        }
        try {
            $Response = Invoke-WebRequest @Params
        }
        catch [System.Net.Http.HttpRequestException], [System.Net.WebException] {
            Die "GET request to metadata attest url ${AttestUrl} failed: $( $_.Exception.Message )" -ExitCode 121
        }
        return ConvertFrom-Json $Response.Content
    }
    return @{
        ActivationUrl   = "${PlatformServicesBaseUrl}/instance/azure/activate"
        Name            = "azure"
         "**********"        = $Function: "**********"
        IsAgentDisabled = $Function:IsAgentDisabled
    }
}

function DedicatedPlatform {
    function GetToken {
        $TokenFile = $Env: "**********"
        if (!(Test-Path -Path $TokenFile -PathType Leaf)) {
            Die "Token file not found: "**********"
        }
        $Token = "**********"
        if (!$Token) {
            Die "Token file is empty: "**********"
        }
        return $Token
    }
    function IsAgentDisabled {
        WriteLog "Platform 'dedicated' does not support bypassing agent bootstrap using tags"  | Write-Debug
        return $false
    }
    return @{
        ActivationUrl   = "${PlatformServicesBaseUrl}/instance/activate"
        Name            = "dedicated"
         "**********"        = $Function: "**********"
        IsAgentDisabled = $Function:IsAgentDisabled
    }
}

function GcpPlatform {
    function GetToken {
        $InstanceMetadataBaseUrl = "http://metadata.google.internal/computeMetadata/v1/instance"
        $Url = "${InstanceMetadataBaseUrl}/service-accounts/default/identity?audience=platform.manage.rackspace.com&format=full"
        $Params = @{
            Uri             = $Url
            Headers         = @{ "Metadata-Flavor" = "Google" }
            UseBasicParsing = $true
        }
        Try {

            $Response = Invoke-WebRequest @Params
        }
        catch [System.Net.Http.HttpRequestException], [System.Net.WebException] {
            Die "GET request to instance identity url ${Url} failed: $( $_.Exception.Message )" -ExitCode 124
        }
        return $Response.Content
    }
    function IsAgentDisabled {
        WriteLog "Checking GCP metadata for ${IgnoreTagKey}" | Write-Debug
        return IsTruthy(GetMetadataAttribute($IgnoreTagKey))
    }
    function GetMetadataAttribute([Parameter(Mandatory = $true)][string]$Key) {
        $InstanceMetadataBaseUrl = "http://metadata.google.internal/computeMetadata/v1/instance"
        $Url = "${InstanceMetadataBaseUrl}/attributes/${Key}"
        $Params = @{
            Uri             = $Url
            Headers         = @{ "Metadata-Flavor" = "Google" }
            UseBasicParsing = $true
        }
        Try {
            $Response = Invoke-WebRequest @Params
            return $Response.Content
        }
        catch [System.Net.WebException] {
            if ($_.Exception.Response.StatusCode -eq 404) {
                return $null
            }
            Die "GET request to instance identity url ${Url} failed: $( $_.Exception.Message )" -ExitCode 125
        }
        catch [System.Net.Http.HttpRequestException] {
            Die "GET request to instance identity url ${Url} failed: $( $_.Exception.Message )" -ExitCode 125
        }
    }
    return @{
        ActivationUrl   = "${PlatformServicesBaseUrl}/instance/gcp/activate"
        Name            = "gcp"
         "**********"        = $Function: "**********"
        IsAgentDisabled = $Function:IsAgentDisabled
    }
}

function GetPlatform([Parameter(Mandatory = $true)][string][ValidateSet("Dedicated", "VMWare", "GCP", "Azure")]$PlatformName) {
    if ($PlatformName -eq 'GCP') {
        return GcpPlatform
    }
    elseif ($PlatformName -eq 'Azure') {
        return AzurePlatform
    }
    elseif ($PlatformName -eq 'Vmware') {
        return VmwarePlatform
    }
    elseif ($PlatformName -eq 'Azure') {
        return AzurePlatform
    }
}

function IsSsmPackageInstalled {
    try {
        Get-Package -Name $AgentPackageName -ProviderName msi
        return $true
    }
    catch {
        return $false
    }
}

function DownloadInstaller(
    [Parameter(Mandatory = $false)][AllowNull()][string]$HttpProxy,
    [Parameter(Mandatory = $false)][AllowNull()][string]$InstallerDownloadRegion) {
    $TmpDir = $env:TEMP + "\ssm"
    New-Item -ItemType Directory -Path $TmpDir -Force | Out-Null
    $OutFile = "${TmpDir}\AmazonSSMAgentSetup.exe"
    if (Test-Path -Path $OutFile -PathType Leaf) {
        WriteLog "Found existing ssm package installer ${OutFile}" | Write-Debug
        return $OutFile
    }
    WriteLog "Using installer download region: ${InstallerDownloadRegion}" | Write-Debug
    $AgentPackageUrl = $DefaultAgentPackageUrl
    if ($InstallerDownloadRegion) {
        $AgentPackageUrl = $RegionalAgentPackageUrl -f $InstallerDownloadRegion
    }
    $Params = @{
        Uri             = $AgentPackageUrl
        OutFile         = $OutFile
        UseBasicParsing = $true
    }
    if ($HttpProxy) {
        WriteLog "Using proxy for connection:  ${HttpProxy}" | Write-Debug
        $Params['Proxy'] = $HttpProxy
    }
    WriteLog "Downloading agent package installer from ${AgentPackageUrl} to ${OutFile}" | Write-Debug
    Try {
        Invoke-WebRequest @Params
    }
    Catch [System.Net.Http.HttpRequestException], [System.Net.WebException] {
        Die "Downloading agent package installer from ${AgentPackageUrl} failed: $( $_.Exception.Message )" -ExitCode 126
    }
    return $OutFile
}

function InstallSsmPackage([Parameter(Mandatory = $true)][string]$PackageFile) {
    $InstallOptions = @("/norestart", "/q", "/log", "install.log")
    WriteLog "Installing agent package: ${PackageFile} $( $InstallOptions -Join " " )" | Write-Debug
    Start-Process $PackageFile -ArgumentList $InstallOptions -Wait
    if (!(IsSsmPackageInstalled)) {
        Die "Package installer failed, check install.log for details" -ExitCode 100
    }
}

function UninstallSsmPackage([Parameter(Mandatory = $false)][AllowNull()][string]$HttpProxy) {
    # Note: uninstalling via Uninstall-Package still leaves the ProviderName=Programs
    # Get-Package -Name $AgentPackageName | Uninstall-Package -AllVersions
    $UninstallOptions = @("/norestart", "/uninstall", "/q", "/log", "uninstall.log")
    $PackageFile = DownloadInstaller -HttpProxy $HttpProxy
    WriteLog "Uninstalling agent package: ${PackageFile} $( $UninstallOptions -Join " " )" | Write-Debug
    Start-Process $PackageFile -ArgumentList $UninstallOptions -Wait
    if (IsSsmPackageInstalled) {
        Die "Package uninstaller failed, check uninstall.log for details" -ExitCode 101
    }
}

function StartSsmService {
    Try {
        $Service = Get-Service -Name $SsmService
        if ($Service.Status -eq 'Running') {
            return $Service
        }
        WriteLog "Starting agent service: ${SsmService}" | Write-Debug
        Start-Service -Name $SsmService
        return CheckSsmServiceStatus('Running')
    }
    Catch {
        Die "Start agent service command failed: $( $_.Exception.Message )" -ExitCode 102
    }
}

function StopSsmService {
    Try {
        $Service = Get-Service -Name $SsmService
        if ($Service.Status -eq 'Stopped') {
            return $Service
        }
        if ($Service.Status -eq 'StartPending') {
            WriteLog "Agent service is in 'StartPending' status, stopping process: $SsmService" | Write-Debug
            Stop-Process -Name $SsmProcess -Force
            Start-Sleep -Seconds 2
        }
        WriteLog "Stopping agent service: ${SsmService}" | Write-Debug
        Stop-Service -Name $SsmService
        return CheckSsmServiceStatus('Stopped')
    }
    Catch {
        Die "Stop agent service command failed: $( $_.Exception.Message )" -ExitCode 103
    }
}

function CheckSsmServiceStatus([Parameter(Mandatory = $true)][string]$Status) {
    $Attempts = 10
    $Delay = 2
    foreach ($Attempt in 1..$Attempts) {
        WriteLog "Checking that agent service is in ${Status} status (attempt: ${Attempt})" | Write-Debug
        $Service = Get-Service -Name $SsmService
        if ($Service.Status -eq $Status) {
            break
        }
        WriteLog "Agent service status is: ${Service.Status}, checking again in ${Delay} seconds" | Write-Debug
        Start-Sleep -Seconds $Delay
    }
    if ($Service.Status -eq $Status) {
        WriteLog "Agent service has reached ${Status} status" | Write-Debug
        return $Service
    }
    Die "Agent service has not reached a ${Status} status after ${Attempts} attempts. Last status: ${Service.Status}" -ExitCode 104
}


function GetJob([Parameter(Mandatory = $true)][string]$JobUrl,
    [Parameter(Mandatory = "**********"
    [Parameter(Mandatory = $false)][AllowNull()][string]$HttpProxy) {

    function FormatJob([Parameter(Mandatory = $true)][PSObject]$Job) {
        return "Job Details:`r`n$( ConvertTo-Json -InputObject $Job -Compress)"
    }

    $GetJobDelay = 5
    $GetJobAttempts = 30
    $Params = @{
        Uri             = $JobUrl
        Method          = 'GET'
        Headers         = "**********"= $Token; 'Accept' = 'application/json' }
        UseBasicParsing = $true
    }
    if ($HttpProxy) {
        WriteLog "Using proxy for connection:  ${HttpProxy}" | Write-Debug
        $Params['Proxy'] = $HttpProxy
    }

    foreach ($Attempt in 1..$GetJobAttempts) {
        WriteLog "Requesting agent activation job (attempt: ${Attempt}): ${JobUrl}" | Write-Debug
        Try {
            $Response = Invoke-WebRequest @Params
        }
        Catch [System.Net.Http.HttpRequestException], [System.Net.WebException] {
            $Message = "GET request to job url ${JobUrl} failed: $( $_.Exception.Message )."
            if ($_.Exception.Response) {
                $Message = "${Message}`r`n$( DumpResponse $_.Exception.Response )"
            }
            Die $Message -ExitCode 105
        }
        $ResponseContent = ConvertFrom-Json $Response.Content
        $JobData = $ResponseContent.data
        $Job = $JobData.items[0]
        if ($Job.status -eq "RUNNING") {
            WriteLog "Agent activation job ${JobUrl} is still running, checking again in ${GetJobDelay} seconds" | Write-Debug
            Start-Sleep -Seconds $GetJobDelay
        }
        elseif ($Job.status -eq "SUCCEEDED") {
            WriteLog "Agent activation job ${JobUrl} completed successfully.`r`n$( FormatJob $Job )" | Write-Debug
            break
        }
        elseif ($Job.status -ne "SUCCEEDED") {
            Die "Agent activation job ${JobUrl} completed unsuccessfully (status: $( Job.status )).`r`n$( FormatJob $Job )" -ExitCode 106
        }
    }
    if ($Job.status -eq "SUCCEEDED") {
        return $Job
    }
    Die "Agent activation job ${JobUrl} did not complete after ${GetJobAttempts}.`r`n`Last $( FormatJob $Job )" -ExitCode 107
}

function DumpResponse([Parameter(Mandatory = $true)][System.Net.HttpWebResponse]$Response) {
    $Data = @{
        StatusCode = $Response.StatusCode
        Headers    = $Response.Headers
        Content    = $Response.Content
    }
    return "Response:`r`n$( ConvertTo-Json -InputObject $Data -Compress)"
}

function GetActivation([Parameter(Mandatory = $true)][hashtable]$Platform,
    [Parameter(Mandatory = $false)][AllowNull()][string]$HttpProxy) {
    WriteLog "Getting auth token for agent activation on platform: "**********"
    $Token = "**********"
    WriteLog "Requesting agent activation: $( $Platform.ActivationUrl )" | Write-Debug
    $Params = @{
        Uri             = $Platform.ActivationUrl
        Method          = 'POST'
        Headers         = "**********"= $Token; 'Content-Type' = 'application/json'; 'Accept' = 'application/json' }
        UseBasicParsing = $true
    }
    if ($HttpProxy) {
        WriteLog "Using proxy for connection:  ${HttpProxy}" | Write-Debug
        $Params['Proxy'] = $HttpProxy
    }
    Try {
        $Response = Invoke-WebRequest @Params
    }
    Catch [System.Net.Http.HttpRequestException], [System.Net.WebException] {
        $Message = "POST request to activation url $( $Platform.ActivationUrl ) failed: $( $_.Exception.Message )."
        if ($_.Exception.Response) {
            $Message = "${Message}`r`n$( DumpResponse $_.Exception.Response )"
        }
        Die $Message -ExitCode 108
    }
    if (!($Response.Headers["Location"])) {
        Die "POST request to activation url $( $Platform.ActivationUrl ) did not return a Location header.`r`n$( DumpResponse $Response )" -ExitCode 109
    }
    $JobUrl = $Response.Headers["Location"]
    WriteLog "Agent activation request responded with job: ${JobUrl}" | Write-Debug
    $Job = "**********"
    return @{
        Code          = $Job.message.activation_code
        Id            = $Job.message.activation_id
        Region        = $Job.message.region
        SystemAccount = $Job.message.system_account
    }
}

function RegisterSsm([Parameter(Mandatory = $true)][hashtable]$Activation) {
    $AgentExePath = "$AgentProgramDir\amazon-ssm-agent.exe"
    try {
        Start-process $AgentExePath -ArgumentList @("-register", "-code", $Activation.Code, "-id", $Activation.Id, "-region", $Activation.Region) -Wait
    }
    catch {
        Die "Agent registration failed: $( $_.Exception.Message )" -ExitCode 110
    }
}

function ActivateSsm([Parameter(Mandatory = $true)][hashtable]$Activation) {
    StopSsmService | Out-Null
    RegisterSsm -Activation $Activation
    StartSsmService | Out-Null
}

function GetSsmDiagnostics {
    $SsmCliPath = "$AgentProgramDir\ssm-cli.exe"
    if (!(Test-Path -Path $SsmCliPath)) {
        Die "${SsmCliPath} command does not exist" -ExitCode 129
    }

    WriteLog "Running agent diagnostics command: ${SsmCliPath} get-diagnostics" | Write-Debug
    $Result = & $SsmCliPath get-diagnostics | Out-String
    if ($?) {
        return ($Result | ConvertFrom-JSON).DiagnosticsOutput
    }
    Die "Agent diagnostics command failed: ${Result}" -ExitCode 111
}

function CheckSsmActivation {
    $SsmDiagnostics = GetSsmDiagnostics
    $FailedChecks = $SsmDiagnostics | Where-Object { $_.Status -eq "Failed" }
    $AgentInfo = GetAgentInformation
    if ($FailedChecks) {
        WriteLog -Level 'WARN' "Some management agent diagnostics failed, please review:`r`n$( ConvertTo-Json -InputObject $FailedChecks -Compress)" | Write-Warning
    }
    if ($AgentInfo) {
        Success "Management agent was registered successfully." -Details $( ConvertTo-Json -InputObject $AgentInfo -Compress )
    }
    else {
        $ErrorLogsPath = "${AgentDataDir}/Logs/errors.log"
        $ErrorLogs = Get-Content $ErrorLogsPath -Tail 10
        Die "Management agent was not registered successfully, display last 10 lines of ${ErrorLogsPath}:`r`n${ErrorLogs}" -ExitCode 112
    }
}

function GetAgentInformation {
    $SsmCliPath = "$AgentProgramDir\ssm-cli.exe"
    if (!(Test-Path -Path $SsmCliPath)) {
        Die "${SsmCliPath} command does not exist" -ExitCode 113
    }

    WriteLog "Running agent instance information command: ${SsmCliPath} get-instance-information" | Write-Debug
    $Result = & $SsmCliPath get-instance-information | Out-String
    if ($?) {
        return ($Result | ConvertFrom-Json)
    }
    $ErrorLine = $Result.Split([Environment]::NewLine) | ForEach-Object {
        if ($_ -match 'error:') {
            return $_
        }
    } | Select-Object -First 1
    WriteLog "Agent is not registered: '${ErrorLine}'" | Write-Debug
    return $false
}

function ClearAgentRegistration {
    $SsmAgentPath = "${AgentProgramDir}\amazon-ssm-agent.exe"
    if (!(Test-Path -Path $SsmAgentPath)) {
        Die "${SsmAgentPath} command does not exist" -ExitCode 127
    }
    WriteLog "Running agent registration clear command: ${SsmAgentPath} -register -clear" | Write-Debug
    $Result = & $SsmAgentPath -register -clear
    if ($?) {
        WriteLog "Successfully cleared agent registration" | Write-Debug
        return $true
    }
    Die "Agent registration clear command failed: ${Result}" -ExitCode 114
}

function CheckOsSupported {
    $ProductType = Get-CimInstance Win32_OperatingSystem | Select-Object -expand ProductType
    $IsServer = $ProductType -eq 3
    $IsDomainController = $ProductType -eq 2
    $OsName = Get-CimInstance Win32_OperatingSystem | Select-Object -expand Caption
    $IsVersionSupported = $null -ne ($SupportedWindowsVersions | Where-Object { $OsName -match $_ })

    if (!($IsServer -or $IsDomainController)) {
        Die "The Rackspace management agent only supports Windows Server or Windows Domain Controllers, it does not support product type: ${ProductType}." -ExitCode 133
    }
    if (!($IsVersionSupported)) {
        Die "The Rackspace management agent supports Windows versions [$( $SupportedWindowsVersions -join "," )], it does not support ${OsName}." -ExitCode 134
    }
    if (!([Environment]::Is64BitOperatingSystem)) {
        Die "The Rackspace management agent only supports 64-bit versions of Windows." -ExitCode 135
    }
}


function CheckPwshVersion {
    if ($PSVersionTable.PSVersion.Major -lt 3) {
        Die "This script requires powershell version > = 3" -ExitCode 143
    }
}
function ConfigureProxy([parameter(Mandatory = $true)][string]$HttpProxy) {
    $ServiceKey = "HKLM:\SYSTEM\CurrentControlSet\Services\AmazonSSMAgent"
    $KeyInfo = (Get-Item -Path $ServiceKey).GetValue("Environment")
    $ProxyVariables = @("http_proxy=${HttpProxy}", "https_proxy=${HttpProxy}", "no_proxy=169.254.169.254")

    if ($null -eq $KeyInfo) {
        New-ItemProperty -Path $ServiceKey -Name Environment -Value $ProxyVariables -PropertyType MultiString -Force
    }
    else {
        Set-ItemProperty -Path $ServiceKey -Name Environment -Value $ProxyVariables
    }
}

function Install(
    [Parameter(Mandatory = $true)][string][ValidateSet("Dedicated", "VMWare", "GCP", "Azure")]$PlatformName,
    [Parameter(Mandatory = $false)][AllowNull()][string]$HttpProxy,
    [Parameter(Mandatory = $false)][AllowNull()][string]$InstallerDownloadRegion
) {
    $Platform = GetPlatform -PlatformName $PlatformName
    WriteLog "Executing install command on platform: $( $Platform.Name )" | Write-Debug
    Try {
        if (&$Platform.IsAgentDisabled) {
            WriteLog "Found a truthy value for '${IgnoreTagKey}' in $( $Platform.Name ) metadata, skipping agent installation" | Write-Debug
            Exit 0
        }
        else {
            WriteLog "Agent install is enabled, '${IgnoreTagKey}' not found in metadata" | Write-Debug
        }
        #if (!(IsSsmPackageInstalled)) {
            $PackageFile = DownloadInstaller -HttpProxy $HttpProxy -InstallerDownloadRegion $InstallerDownloadRegion
            InstallSsmPackage -PackageFile $PackageFile
        #}
        if ($HttpProxy) {
            WriteLog "Using proxy for connection:  ${HttpProxy}" | Write-Debug
            ConfigureProxy -HttpProxy $HttpProxy
        }
        $Registration = GetAgentInformation
        if ($Registration) {
            WriteLog "Agent is already registered $( ConvertTo-Json -InputObject $Registration -Compress)" | Write-Debug
            StartSsmService | Out-Null
            CheckSsmActivation
            Exit 0
        }

        $Activation = GetActivation -Platform $Platform -HttpProxy $HttpProxy
        ActivateSsm -Activation $Activation
        WriteLog "Waiting ${RegistrationWait} seconds before checking agent activation status" | Write-Debug
        Start-Sleep -Seconds $RegistrationWait
        CheckSsmActivation
    }
    Catch [System.Net.Http.HttpRequestException], [System.Net.WebException] {
        Die "HTTP request failed: $( $_.Exception.Message )" -ExitCode 115
    }
}

function Reregister([Parameter(Mandatory = $true)][string][ValidateSet("Dedicated", "VMWare", "GCP", "Azure")]$PlatformName,
    [Parameter(Mandatory = $false)][AllowNull()][string]$HttpProxy) {
    WriteLog "Executing reregister command on platform ${PlatformName}" | Write-Debug
    $Platform = GetPlatform -PlatformName $PlatformName
    Try {
        if (&$Platform.IsAgentDisabled) {
            WriteLog "Found a truthy value for '${IgnoreTagKey}' in $( $Platform.Name ) metadata, skipping agent installation" | Write-Debug
            Exit 0
        }
        else {
            WriteLog "Agent install is enabled, '${IgnoreTagKey}' not found in metadata" | Write-Debug
        }
        if (!(IsSsmPackageInstalled)) {
            Die "SSM package is not installed, cannot reregister agent" -ExitCode 128
        }
        StopSsmService | Out-Null
        ClearAgentRegistration | Out-Null
        $Activation = GetActivation -Platform $Platform -HttpProxy $HttpProxy
        ActivateSsm -Activation $Activation
        WriteLog "Waiting ${RegistrationWait} seconds before checking agent activation status" | Write-Debug
        Start-Sleep -Seconds $RegistrationWait
        CheckSsmActivation
    }
    Catch [System.Net.Http.HttpRequestException], [System.Net.WebException] {
        Die "HTTP request failed: $( $_.Exception.Message )" -ExitCode 117
    }
}

function Uninstall([Parameter(Mandatory = $false)][AllowNull()][string]$HttpProxy) {
    WriteLog "Executing uninstall command" | Write-Debug
    if (!(IsSsmPackageInstalled)) {
        Die "Package $SsmService is not currently installed" -ExitCode 116
    }
    StopSsmService
    ClearAgentRegistration | Out-Null
    UninstallSsmPackage -HttpProxy $HttpProxy
    Success "Agent was successfully uninstalled"
}

function Main {
    Param (
        [Parameter(Mandatory = $true)][string]$Platform,
        [Parameter(Mandatory = $true)][string]$Command,
        [Parameter(Mandatory = $false)][string]$HttpProxy,
        [Parameter(Mandatory = $false)][string]$InstallerDownloadRegion
    )
    process {
        Try {
            CheckOsSupported
            CheckPwshVersion

            if ($Command -eq "Install") {
                Install -PlatformName $Platform -HttpProxy $HttpProxy -InstallerDownloadRegion $InstallerDownloadRegion
            }
            elseif ($Command -eq "Uninstall") {
                Uninstall -HttpProxy $HttpProxy
            }
            elseif ($Command -eq "Reregister") {
                Reregister -PlatformName $Platform
            }
        }
        Catch {
            $err = $_
            Write-Error ($err.Exception | Format-List -Force | Out-String) -ErrorAction Continue
            Die "Uncaught exception: [$( $err.Exception.getType().fullname )] $( $err.Exception.Message ), exiting..." -ExitCode 144
        }
    }
}

if ($MyInvocation.InvocationName -ne ".") {
    Write-Output($MyInvocation.InvocationName)
    Main -Command $Command -Platform $Platform -HttpProxy $HttpProxy -InstallerDownloadRegion $InstallerDownloadRegion
}



 $InstallerDownloadRegion
            }
            elseif ($Command -eq "Uninstall") {
                Uninstall -HttpProxy $HttpProxy
            }
            elseif ($Command -eq "Reregister") {
                Reregister -PlatformName $Platform
            }
        }
        Catch {
            $err = $_
            Write-Error ($err.Exception | Format-List -Force | Out-String) -ErrorAction Continue
            Die "Uncaught exception: [$( $err.Exception.getType().fullname )] $( $err.Exception.Message ), exiting..." -ExitCode 144
        }
    }
}

if ($MyInvocation.InvocationName -ne ".") {
    Write-Output($MyInvocation.InvocationName)
    Main -Command $Command -Platform $Platform -HttpProxy $HttpProxy -InstallerDownloadRegion $InstallerDownloadRegion
}



