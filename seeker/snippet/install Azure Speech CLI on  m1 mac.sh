#date: 2022-12-29T16:42:36Z
#url: https://api.github.com/gists/c4b8939ff8fd7c0f0887efb213d0fd01
#owner: https://api.github.com/users/odekopoon

# 
# https://learn.microsoft.com/en-us/azure/cognitive-services/speech-service/spx-basics?tabs=macOS%2Cterminal

# 1. x64むけの.net SDKをダウンロードする
open https://dotnet.microsoft.com/en-us/download
# ".net SDK x64"をダウンロードする
# ".net SDK x64"インストールする


# 2. x64用のdotnetからAzure Speech CLIをインストール
/usr/local/share/dotnet/x64/dotnet tool install --global Microsoft.CognitiveServices.Speech.CLI
# ~/.dotnet/tools

# 3a. spxコマンドで、x64用になる
spx


# 3b. 間違えてarm64のdotnetでインストールしたときは、spxが動かないので、uninstallして入れ直すか、 spx.dllを直接実行する

% ~/.dotnet/tools|grep spx.dll
/Users/XXXXXXXX/.dotnet/tools/.store/microsoft.cognitiveservices.speech.cli/1.24.2/microsoft.cognitiveservices.speech.cli/1.24.2/tools/net6.0/any/spx.dll
/usr/local/share/dotnet/x64/dotnet ~/.dotnet/tools/.store/microsoft.cognitiveservices.speech.cli/1.24.2/microsoft.cognitiveservices.speech.cli/1.24.2/tools/net6.0/any/spx.dll


==

% LANG=utf8.en_us /usr/local/share/dotnet/x64/dotnet --info
.NET SDK (reflecting any global.json):
 Version:   6.0.403
 Commit:    2bc18bf292

Runtime Environment:
 OS Name:     Mac OS X
 OS Version:  13.1
 OS Platform: Darwin
 RID:         osx-x64
 Base Path:   /usr/local/share/dotnet/x64/sdk/6.0.403/

global.json file:
  Not found

Host:
  Version:      6.0.11
  Architecture: x64
  Commit:       943474ca16

.NET SDKs installed:
  6.0.403 [/usr/local/share/dotnet/x64/sdk]

.NET runtimes installed:
  Microsoft.AspNetCore.App 6.0.11 [/usr/local/share/dotnet/x64/shared/Microsoft.AspNetCore.App]
  Microsoft.NETCore.App 6.0.11 [/usr/local/share/dotnet/x64/shared/Microsoft.NETCore.App]

Download .NET:
  https://aka.ms/dotnet-download

Learn about .NET Runtimes and SDKs:
  https://aka.ms/dotnet/runtimes-sdk-info
  
  

% LANG=utf8.en_us /usr/local/share/dotnet/x64/dotnet ~/.dotnet/tools/.store/microsoft.cognitiveservices.speech.cli/1.24.2/microsoft.cognitiveservices.speech.cli/1.24.2/tools/net6.0/any/spx.dll

SPX - Azure Speech CLI, Version 1.24.2
Copyright (c) 2022 Microsoft Corporation. All Rights Reserved.

  ______ ___ _  __
 /  ___// _ \ \/ /
 \___ \/ ___/   <
/____ /_/  /__/\_\

USAGE: spx <command> [...]

HELP

  spx help
  spx help setup

COMMANDS

  spx init [...]            (see: spx help init)
  spx config [...]          (see: spx help config)

  spx recognize [...]       (see: spx help recognize)
  spx synthesize [...]      (see: spx help synthesize)

  spx intent [...]          (see: spx help intent)
  spx translate [...]       (see: spx help translate)

  spx batch [...]           (see: spx help batch)
  spx csr [...]             (see: spx help csr)

  spx profile [...]         (see: spx help profile)
  spx speaker [...]         (see: spx help speaker)

  spx webjob [...]          (see: spx help webjob)

ADDITIONAL TOPICS

  spx help examples

  spx help find "mp3"
  spx help find "mp3" --expand

  spx help find topics "examples"
  spx help list topics

  spx help documentation

