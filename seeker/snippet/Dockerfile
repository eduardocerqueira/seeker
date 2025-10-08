#date: 2025-10-08T17:12:29Z
#url: https://api.github.com/gists/dfedfc1927c9245c67835ad1642e9504
#owner: https://api.github.com/users/ababushk

FROM mcr.microsoft.com/windows/servercore:ltsc2022

# Install Visual Studio Build Tools
# Inspired by https://github.com/microsoft/vs-dockerfiles/tree/main/native-desktop and
# https://learn.microsoft.com/en-us/visualstudio/install/advanced-build-tools-container?view=vs-2022

# Pin Build Tools version to avoid surprises after image re-builds.
# Links to specific versions of Build Tools are available here: https://learn.microsoft.com/en-us/visualstudio/releases/2022/release-history
ENV BUILDTOOLS_URI="https://download.visualstudio.microsoft.com/download/pr/286c67ca-51f4-409d-ade8-3036a5184667/a8a9a3b82f278f504156a940dcfd5619e9f214eb7e9071c5f5571a0f8baa94f3/vs_BuildTools.exe"

# .vsconfig can be exported from your Windows laptop's Build Tools installation
# (there's an option in the GUI).
COPY .vsconfig C:\TEMP\.vsconfig

# If image build fails on BuildTools installation, do:
# docker cp <intermediate_container_id>:C:\vslogs.zip .\vslogs.zip
# to study the logs
COPY Install-BuildTools.ps1 C:\TEMP\Install-BuildTools.ps1
RUN pwsh -Command "Set-MpPreference -DisableRealtimeMonitoring $true"; `
    C:\TEMP\Install-BuildTools.ps1 -InstallPath C:\BuildTools `
    -VsConfigPath C:\TEMP\.vsconfig `
    -LogsPath C:\vslogs.zip `
    -BuildToolsUri $Env:BUILDTOOLS_URI