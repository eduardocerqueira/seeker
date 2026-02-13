#date: 2026-02-13T17:27:54Z
#url: https://api.github.com/gists/094991511786c4d0e639524e10fe2317
#owner: https://api.github.com/users/kurtwuckertjr

#!/bin/bash
# Clone all b-open-io repos into categorized directories
# Run this on Clurt's VM

set -e

BASE="$HOME/.openclaw/workspace/repos"
mkdir -p "$BASE"/{sdks,tools,overlays,auth,wallet,infra,apps,tokens,agents,websites,media,misc}

clone() {
  local category="$1"
  local repo="$2"
  local dir="$BASE/$category/$(basename "$repo")"
  if [ -d "$dir" ]; then
    echo "  [skip] $repo already cloned"
  else
    echo "  [clone] $repo -> $category/"
    gh repo clone "$repo" "$dir" -- --depth 1 2>/dev/null || echo "  [FAIL] $repo"
  fi
}

echo "=== SDKs ==="
clone sdks b-open-io/ts-sdk
clone sdks b-open-io/go-sdk
clone sdks b-open-io/1sat-sdk
clone sdks b-open-io/run-sdk
clone sdks b-open-io/ts-templates
clone sdks b-open-io/go-templates
clone sdks b-open-io/satoshi-token
clone sdks b-open-io/wallet-toolbox
clone sdks b-open-io/1sat-wallet-toolbox
clone sdks b-open-io/spv-store
clone sdks b-open-io/bitcoinschema
clone sdks b-open-io/theme-token-sdk
clone sdks b-open-io/txex

echo "=== Overlays ==="
clone overlays b-open-io/overlay
clone overlays b-open-io/bsv21-overlay
clone overlays b-open-io/opns-overlay
clone overlays b-open-io/a2b-overlay
clone overlays b-open-io/bap-overlay
clone overlays b-open-io/bsocial-overlay
clone overlays b-open-io/gib-overlay
clone overlays b-open-io/go-overlay-fiber
clone overlays b-open-io/go-overlay-discovery-services
clone overlays b-open-io/go-overlay-services
clone overlays b-open-io/bsv21-overlay-1sat-sync
clone overlays b-open-io/1sat-stack

echo "=== Auth ==="
clone auth b-open-io/bitcoin-auth
clone auth b-open-io/go-bitcoin-auth
clone auth b-open-io/sigma-auth
clone auth b-open-io/sigma-auth-web
clone auth b-open-io/better-auth-plugin
clone auth b-open-io/AIP
clone auth b-open-io/sigmaidentity
clone auth b-open-io/sigmaidentity.com

echo "=== Wallet ==="
clone wallet b-open-io/pay-purse
clone wallet b-open-io/bitcoin-backup
clone wallet b-open-io/tokenpass-server
clone wallet b-open-io/tokenpass.app
clone wallet b-open-io/tokenpass-desktop

echo "=== Tools ==="
clone tools b-open-io/bsv-mcp
clone tools b-open-io/claude-peacock
clone tools b-open-io/claude-plugins
clone tools b-open-io/claude-perms
clone tools b-open-io/statusline
clone tools b-open-io/vscode-bitcoin
clone tools b-open-io/bitbench
clone tools b-open-io/scribe
clone tools b-open-io/scribe-desktop
clone tools b-open-io/pr-agent
clone tools b-open-io/bap-cli
clone tools b-open-io/bsocial-cli
clone tools b-open-io/pow20-miner
clone tools b-open-io/gib
clone tools b-open-io/prompts
clone tools b-open-io/homebrew-tap
clone tools b-open-io/winget-pkgs
clone tools b-open-io/theme-token-cli

echo "=== Agents / ClawNet ==="
clone agents b-open-io/agent-master
clone agents b-open-io/agent-master-engine
clone agents b-open-io/agent-master-cli
clone agents b-open-io/clawbook.network
clone agents b-open-io/clawbook-bot
clone agents b-open-io/openclaw-bot
clone agents b-open-io/clawnet
clone agents b-open-io/clawnet-bot
clone agents b-open-io/moltbot-sandbox
clone agents b-open-io/clawbook-skills
clone agents b-open-io/bopen-ai

echo "=== Infra / Skills ==="
clone infra b-open-io/bsv-skills
clone infra b-open-io/1sat-skills
clone infra b-open-io/product-skills
clone infra b-open-io/gemskills
clone infra b-open-io/ordfs-server
clone infra b-open-io/go-ordfs-server
clone infra b-open-io/block-headers-service
clone infra b-open-io/gorillanode
clone infra b-open-io/go-junglebus
clone infra b-open-io/junglebus
clone infra b-open-io/go-faucet-api
clone infra b-open-io/go-bmap-indexer
clone infra b-open-io/bmap-api

 "**********"e "**********"c "**********"h "**********"o "**********"  "**********"" "**********"= "**********"= "**********"= "**********"  "**********"T "**********"o "**********"k "**********"e "**********"n "**********"s "**********"  "**********"= "**********"= "**********"= "**********"" "**********"
clone tokens b-open-io/theme-token
clone tokens b-open-io/theme-token-preview-harness
clone tokens b-open-io/mintflow
clone tokens b-open-io/mnee-cosigner
clone tokens b-open-io/mnee-dashboard
clone tokens b-open-io/mnee-ordinals-service
clone tokens b-open-io/mnee-ordinals-contract

echo "=== Media ==="
clone media b-open-io/bitcoin-image
clone media b-open-io/go-bitcoin-image
clone media b-open-io/jamify
clone media b-open-io/jamify-hls
clone media b-open-io/a2b
clone media b-open-io/droplit
clone media b-open-io/bitcoin-asset
clone media b-open-io/tts-api

echo "=== Websites ==="
clone websites b-open-io/1sat-website
clone websites b-open-io/nodeless-website
clone websites b-open-io/bigblocks.dev
clone websites b-open-io/bigblocks
clone websites b-open-io/bigblocks-registry
clone websites b-open-io/openprotocollabs.com
clone websites b-open-io/gorillapool.com
clone websites b-open-io/1satordinals.com
clone websites b-open-io/bopen.io
clone websites b-open-io/metalens.app
clone websites b-open-io/1sat-university
clone websites b-open-io/scribe-promo

echo "=== Apps ==="
clone apps b-open-io/bitchat-nitro
clone apps b-open-io/rwa-pro
clone apps b-open-io/1sat-name
clone apps b-open-io/bottlebird
clone apps b-open-io/HyperSwag
clone apps b-open-io/opl-admin
clone apps b-open-io/alchema-genesis
clone apps b-open-io/alchema-story

echo "=== Misc ==="
clone misc b-open-io/demo-repository
clone misc b-open-io/roadmap
clone misc b-open-io/design-productive
clone misc b-open-io/ordi-training
clone misc b-open-io/bitpic.net
clone misc b-open-io/minerva
clone misc b-open-io/.github
clone misc b-open-io/.github-private
clone misc b-open-io/bitchat-nitro

echo ""
echo "=== Done! ==="
echo "Repos cloned to: $BASE"
find "$BASE" -mindepth 2 -maxdepth 2 -type d | wc -l | xargs -I{} echo "Total repos: {}"
