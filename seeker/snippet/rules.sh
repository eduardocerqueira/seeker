#date: 2025-12-18T17:13:39Z
#url: https://api.github.com/gists/061807901c25bb29bf2c40ea28a8c0ef
#owner: https://api.github.com/users/yunuo1025

#!/bin/bash

# Note:
# When the script prompts you to provide the dedicated configuration link, please copy and use the RAW link of your own Gist!
# 
# 注意：
# 当脚本提示你提供专属配置链接时，请使用你自己的Gist的RAW链接！



# Required.
#
# If you can't access GitHub directly (e.g., due to network restrictions), set a proxy URL prefix here. 
# Note that the prefix will be added to the beginning of all your rule URLs as well.
#
# Otherwise, LEAVE IT BLANK!!!
#
# Example:
# GLOBAL_GITHUB_PROXY_URL="https://my_github_proxy_url_prefix.com"
GLOBAL_GITHUB_PROXY_URL=""



# Required.
#
# Modify this parameter if the subscription process fails. (See https://github.com/immortalwrt/homeproxy/pull/189 for more information.)
#
# Otherwise, LEAVE IT BLANK!!!
#
# Example: SUBSCRIPTION_USER_AGENT="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
SUBSCRIPTION_USER_AGENT=""



# Optional.
#
# If defined, the script will invoke the embedded homeproxy subscription script to automatically complete the subscription process for your proxy service(s).
SUBSCRIPTION_URLS=(
  # Change to your own subscription URL(s).
  "https://abc.com?subscribe=123#Your_custom_proxy_server_name_01"
  "https://xyz.com?subscribe=456#Your_custom_proxy_server_name_02"
)



# Required.
RULESET_URLS=(
  #
  # "Your_Node_Name|
  # URL1
  # URL2
  # URL3
  # /absolute/file/path/file1.json
  # /absolute/file/path/file2.srs
  # ..."
  #

  # Optional: Delete the entire 'reject_out' definition if you don't need any ad rules.
  "reject_out|
  https://raw.githubusercontent.com/privacy-protection-tools/anti-ad.github.io/master/docs/anti-ad-sing-box.srs"
  
  #
  #
  #  -----------------  Rule-Sets start -----------------
  #
  #

  "PROXY_SERVER_01_US|
  /etc/homeproxy/ruleset/MyProxy.json
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/google@cn.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/google-gemini.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/google-trust-services.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/google-trust-services@cn.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/google-play.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/google-play@cn.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/googlefcm.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/google.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geoip/google.srs"

  "PROXY_SERVER_02_US|
  /etc/homeproxy/ruleset/MyAI.json
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/openai.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/bing.srs
  https://raw.githubusercontent.com/KaringX/karing-ruleset/sing/geo/geoip/ai.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/telegram.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geoip/telegram.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/discord.srs"
  
  "PROXY_SERVER_02_SG_With_Or_Without_Suffix|
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/twitch.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/amazon.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/amazon@cn.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/amazontrust.srs"
  
  "PROXY_SERVER_02_US_IPv6_AsBackup|
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/twitter.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/x.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/sing/geo/geoip/twitter.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/tiktok.srs"
 
  # Optional: Delete the entire 'direct_out' definition if you don't need any domestic rules.
  "direct_out|
  /etc/homeproxy/ruleset/MyDirect.json
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/microsoft@cn.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/azure@cn.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/apple-cn.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geoip/cn.srs
  https://raw.githubusercontent.com/MetaCubeX/meta-rules-dat/refs/heads/sing/geo/geosite/cn.srs"
  
  #
  #
  #  -----------------  Rule-Sets end -----------------
  #
  #

)



# Required.
DNS_SERVERS=(
  # "Your_DNS_Server_Name|
  # DoH
  # DoT
  # UDP
  # See https://sing-box.sagernet.org/configuration/dns/server for more info.
  # ...
  # "
  #
  
  "PROXY_SERVER_01_US|
  https://dns.google/dns-query"
  
  "PROXY_SERVER_02_US|
  https://dns.google/dns-query"
  
  "PROXY_SERVER_02_SG_With_Or_Without_Suffix|
  https://1.1.1.1/dns-query"
  
  "PROXY_SERVER_02_US_IPv6_AsBackup|
  2001:4860:4860:0000:0000:0000:0000:8888"
  
  "Default_DNS_Server|
  https://dns.google/dns-query
  https://cloudflare-dns.com/dns-query
  https://doh.opendns.com/dns-query"
)