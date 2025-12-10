#date: 2025-12-10T17:04:04Z
#url: https://api.github.com/gists/754db22940e32a3e180c288bab8bfa03
#owner: https://api.github.com/users/Kamesuta

#!/usr/bin/env bash
set -euo pipefail

API="https://api.github.com/repos/ViaVersion/ViaProxy/releases/latest"
echo "🔍 ViaProxy の最新版を確認中..." >&2

# 最新リリースの JSON から Java8 を含まない .jar の URL を抜き出す
asset_url="$(
  curl -fsSL "$API" |
    grep -o '"browser_download_url": *"[^"]*ViaProxy-[^"]*\.jar"' |
    grep -v 'java8' |
    head -n1 |
    sed 's/.*"browser_download_url": *"\([^"]*\)".*/\1/'
)"

if [[ -n "$asset_url" ]]; then
  echo "⬇️ ダウンロード開始: $asset_url" >&2
  if curl -# -fL "$asset_url" -o ViaProxy.jar.new; then
    mv ViaProxy.jar.new ViaProxy.jar
    echo "✅ 更新しました: ViaProxy.jar" >&2
  else
    echo "⚠️ ダウンロード失敗。既存バージョンで起動します。" >&2
    rm -f ViaProxy.jar.new
  fi
else
  echo "⚠️ 更新情報の取得に失敗。既存バージョンで起動します。" >&2
fi

# 起動（既存 or 新規）
if [[ -f ViaProxy.jar ]]; then
  echo "🚀 起動中..." >&2
  java -Xms128M -Xmx4G -Dterminal.jline=false -Dterminal.ansi=true -jar ViaProxy.jar config viaproxy.yml
else
  echo "❌ ViaProxy.jar が存在しません。初回取得に失敗しました。" >&2
  exit 1
fi
