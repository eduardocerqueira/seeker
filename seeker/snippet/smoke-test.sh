#date: 2026-03-05T18:33:02Z
#url: https://api.github.com/gists/63f5164e7a44d35575d2d3df995b0e3f
#owner: https://api.github.com/users/daivic

#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# swap-service Smoke Test Suite
# ═══════════════════════════════════════════════════════════════════════════════
# Tests all 51 HTTP routes + 7 gRPC methods against the dev environment.
# Uses real EOA + SCW wallets for proper dual-path testing.
# Validates response bodies contain expected fields, not just status codes.
#
# Usage:
#   ./docs/smoke-test.sh                    # Run all tests
#   ./docs/smoke-test.sh --section swap     # Run only swap section
#   ./docs/smoke-test.sh --verbose          # Show response bodies
#   ./docs/smoke-test.sh --save-baseline    # Save results to docs/baseline.json
#
# Prerequisites:
#   - cb-serviceauth-dev-jwt installed
#   - grpcurl installed (for gRPC tests)
#   - curl installed
#   - jq installed (for response validation)
# ═══════════════════════════════════════════════════════════════════════════════

set -o pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
BOLD='\033[1m'
DIM='\033[2m'
NC='\033[0m'

# ── Config ────────────────────────────────────────────────────────────────────
BASE_URL="${SMOKE_BASE_URL:-https://wallet-api-dev.cbhq.net}"
GRPC_HOST="${SMOKE_GRPC_HOST:-wallet-swap-dev.cbhq.net:9090}"
GRPCURL="/opt/homebrew/bin/grpcurl"
JWT_CMD="/Users/daivicvora/go/bin/cb-serviceauth-dev-jwt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_DIR="${SCRIPT_DIR}/../protos"

# ── Test Wallets ──────────────────────────────────────────────────────────────
# SCW (Smart Contract Wallet) — used for SCW-specific tests
SCW_WALLET="0x855ef1d0e1b0d6bfa67d1f1b42c20fd6c66eafde"
# EOA (Externally Owned Account) — used for EOA-specific tests
EOA_WALLET="0x9036464e4ecD2d40d21EE38a0398AEdD6805a09B"
# Solana wallet (funded with WSOL)
SOL_WALLET="DKCGgPdyLcPFJGTZzkhYeenEUWPu5VED5ourTrW8PAM"

# ── Token Addresses ───────────────────────────────────────────────────────────
ETH_NATIVE="0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
BASE_USDC="0x833589fcd6edb6e08f4c7c32d4f71b54bda02913"
BASE_WETH="0x4200000000000000000000000000000000000006"
ETH_USDC="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
ETH_WETH="0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
SOL_WSOL="So11111111111111111111111111111111111111112"
RETH="0xae78736Cd615f374D3085123A210448E74Fc6393"

# ── Test Amounts (sized UNDER wallet balances) ────────────────────────────────
# SCW: Base ETH 0.000888, USDC 18, ETH mainnet 0.0011
# EOA: Base ETH 0.0876, USDC 66, ETH mainnet 0.0021
# SOL: WSOL 0.044
SMALL_ETH="100000000000000"           # 0.0001 ETH (safe for both wallets)
SMALL_USDC="1000000"                  # 1 USDC
SMALL_SOL="10000000"                  # 0.01 SOL (lamports)

# ── Counters ──────────────────────────────────────────────────────────────────
TOTAL=0
PASSED=0
FAILED=0
EXPECTED_FAILED=0
SKIPPED=0
UNEXPECTED_PASS=0
FAIL_DETAILS=()
UNEXPECTED_DETAILS=()
BASELINE_ENTRIES=()

# ── CLI Args ──────────────────────────────────────────────────────────────────
VERBOSE=false
SECTION_FILTER=""
SAVE_BASELINE=false
prev_arg=""
for arg in "$@"; do
  case "$arg" in
    --verbose|-v)       VERBOSE=true ;;
    --save-baseline)    SAVE_BASELINE=true ;;
    --section)          : ;; # next arg is the section name
    *)
      if [[ "$prev_arg" == "--section" ]]; then
        SECTION_FILTER="$arg"
      fi
      ;;
  esac
  prev_arg="$arg"
done

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  swap-service Smoke Test Suite${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${DIM}  Target:      ${BASE_URL}${NC}"
echo -e "${DIM}  gRPC:        ${GRPC_HOST}${NC}"
echo -e "${DIM}  EOA Wallet:  ${EOA_WALLET}${NC}"
echo -e "${DIM}  SCW Wallet:  ${SCW_WALLET}${NC}"
echo -e "${DIM}  SOL Wallet:  ${SOL_WALLET}${NC}"
echo ""

# ── Auth ──────────────────────────────────────────────────────────────────────
echo -n "Generating auth token... "
 "**********"i "**********"f "**********"  "**********"! "**********"  "**********"T "**********"O "**********"K "**********"E "**********"N "**********"= "**********"$ "**********"( "**********"" "**********"$ "**********"J "**********"W "**********"T "**********"_ "**********"C "**********"M "**********"D "**********"" "**********"  "**********"" "**********"w "**********"a "**********"l "**********"l "**********"e "**********"t "**********"/ "**********"s "**********"w "**********"a "**********"p "**********"- "**********"s "**********"e "**********"r "**********"v "**********"i "**********"c "**********"e "**********": "**********": "**********": "**********": "**********"" "**********"  "**********"2 "**********"> "**********"/ "**********"d "**********"e "**********"v "**********"/ "**********"n "**********"u "**********"l "**********"l "**********") "**********"; "**********"  "**********"t "**********"h "**********"e "**********"n "**********"
  echo -e "${RED}FAILED${NC}"
  echo "Could not generate JWT. Is cb-serviceauth-dev-jwt installed at $JWT_CMD?"
  exit 1
fi
echo -e "${GREEN}OK${NC}"

# ── Prereq checks ────────────────────────────────────────────────────────────
HAS_JQ=false
echo -n "Checking jq... "
if command -v jq &>/dev/null; then
  HAS_JQ=true
  echo -e "${GREEN}OK${NC}"
else
  echo -e "${YELLOW}NOT FOUND${NC} — response validation will be limited"
fi

HAS_GRPCURL=false
echo -n "Checking grpcurl... "
if [[ -x "$GRPCURL" ]]; then
  HAS_GRPCURL=true
  echo -e "${GREEN}OK${NC}"
else
  echo -e "${YELLOW}NOT FOUND${NC} — gRPC tests will be skipped"
fi
echo ""

# ── Temp files ────────────────────────────────────────────────────────────────
TMPBODY=$(mktemp)
trap 'rm -f "$TMPBODY"' EXIT

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

do_curl() {
  local method="$1" url="$2" body="${3:-}"
  shift 3
  local extra_headers=("$@")

  local curl_args=(
    -s -w "%{http_code}"
    -o "$TMPBODY"
    -X "$method"
    -H "Authorization: "**********"
    -H "Content-Type: application/json"
    --connect-timeout 10
    --max-time 30
  )

  for h in ${extra_headers[@]+"${extra_headers[@]}"}; do
    curl_args+=(-H "$h")
  done

  if [[ -n "$body" ]]; then
    curl_args+=(-d "$body")
  fi

  curl "${curl_args[@]}" "$url" 2>/dev/null || echo "000"
}

# Check response body contains expected field names (simple grep — works without jq)
# Usage: check_body_fields "field1" "field2" ...
# Returns 0 if ALL found, 1 if any missing. Prints comma-separated missing list.
check_body_fields() {
  local body
  body=$(cat "$TMPBODY" 2>/dev/null || echo "")
  local missing=()
  for field in "$@"; do
    if ! echo "$body" | grep -q "\"${field}\""; then
      missing+=("$field")
    fi
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "${missing[*]}"
    return 1
  fi
  return 0
}

# Global: set before calling test_http to validate fields in response body
VALIDATE_FIELDS=()

# ── test_http ─────────────────────────────────────────────────────────────────
# Usage: test_http NAME METHOD PATH [BODY] [EXPECTED_STATUS] [EXTRA_HEADERS...]
test_http() {
  local name="$1" method="$2" path="$3" body="${4:-}" expected_status="${5:-200}"
  local _n=$(( $# < 5 ? $# : 5 ))
  shift $_n
  local extra_headers=("$@")

  TOTAL=$((TOTAL + 1))
  local url="${BASE_URL}${path}"
  local status
  status=$(do_curl "$method" "$url" "$body" ${extra_headers[@]+"${extra_headers[@]}"})

  local resp_body resp_preview
  resp_body=$(cat "$TMPBODY" 2>/dev/null || echo "")
  resp_preview="${resp_body:0:200}"

  local fields_to_check=("${VALIDATE_FIELDS[@]+"${VALIDATE_FIELDS[@]}"}")
  VALIDATE_FIELDS=()  # always reset

  if [[ "$status" == "$expected_status" ]]; then
    # Validate response body fields if requested
    if [[ ${#fields_to_check[@]} -gt 0 ]]; then
      local missing
      if missing=$(check_body_fields "${fields_to_check[@]}"); then
        PASSED=$((PASSED + 1))
        echo -e "  ${GREEN}[PASS]${NC} ${name} ${DIM}(${method} ${path})${NC} ${DIM}- ${status}${NC}"
      else
        FAILED=$((FAILED + 1))
        echo -e "  ${RED}[FAIL]${NC} ${name} ${DIM}(${method} ${path})${NC} - ${status} OK but missing: ${missing}"
        FAIL_DETAILS+=("${name}: ${status} but missing fields: ${missing}")
      fi
    else
      PASSED=$((PASSED + 1))
      echo -e "  ${GREEN}[PASS]${NC} ${name} ${DIM}(${method} ${path})${NC} ${DIM}- ${status}${NC}"
    fi
    if [[ "$VERBOSE" == true ]]; then
      echo -e "         ${DIM}${resp_preview}${NC}"
    fi
    BASELINE_ENTRIES+=("{\"name\":\"${name//\"/\\\"}\",\"status\":${status},\"result\":\"pass\"}")
  else
    FAILED=$((FAILED + 1))
    echo -e "  ${RED}[FAIL]${NC} ${name} ${DIM}(${method} ${path})${NC} - Expected ${expected_status}, got ${status}"
    echo -e "         ${DIM}${resp_preview}${NC}"
    FAIL_DETAILS+=("${name}: Expected ${expected_status}, got ${status} | ${resp_preview}")
    BASELINE_ENTRIES+=("{\"name\":\"${name//\"/\\\"}\",\"status\":${status},\"result\":\"fail\"}")
  fi
}

# ── expected_fail ─────────────────────────────────────────────────────────────
expected_fail() {
  local name="$1" reason="$2" method="$3" path="$4" body="${5:-}"
  local _n=$(( $# < 5 ? $# : 5 ))
  shift $_n
  local extra_headers=("$@")

  TOTAL=$((TOTAL + 1))
  local url="${BASE_URL}${path}"
  local status
  status=$(do_curl "$method" "$url" "$body" ${extra_headers[@]+"${extra_headers[@]}"})

  if [[ "$status" == "200" ]]; then
    UNEXPECTED_PASS=$((UNEXPECTED_PASS + 1))
    PASSED=$((PASSED + 1))
    echo -e "  ${MAGENTA}[UNEXPECTED_PASS]${NC} ${name} ${DIM}(${method} ${path})${NC} - Got 200! (was: ${reason})"
    UNEXPECTED_DETAILS+=("${name}: Now returns 200 (was: ${reason})")
    BASELINE_ENTRIES+=("{\"name\":\"${name//\"/\\\"}\",\"status\":200,\"result\":\"unexpected_pass\"}")
  else
    EXPECTED_FAILED=$((EXPECTED_FAILED + 1))
    echo -e "  ${YELLOW}[EXPECTED_FAIL]${NC} ${name} ${DIM}(${method} ${path})${NC} ${DIM}- ${status} (${reason})${NC}"
    BASELINE_ENTRIES+=("{\"name\":\"${name//\"/\\\"}\",\"status\":${status},\"result\":\"expected_fail\"}")
  fi
}

# ── skip ──────────────────────────────────────────────────────────────────────
skip() {
  local name="$1" reason="$2"
  TOTAL=$((TOTAL + 1))
  SKIPPED=$((SKIPPED + 1))
  echo -e "  ${CYAN}[SKIP]${NC} ${name} ${DIM}- ${reason}${NC}"
  BASELINE_ENTRIES+=("{\"name\":\"${name//\"/\\\"}\",\"status\":0,\"result\":\"skip\"}")
}

# ── test_grpc ─────────────────────────────────────────────────────────────────
test_grpc() {
  local name="$1" method="$2" json_data="$3" expect_fail="${4:-false}" fail_reason="${5:-}"

  TOTAL=$((TOTAL + 1))

  if [[ "$HAS_GRPCURL" == false ]]; then
    SKIPPED=$((SKIPPED + 1))
    echo -e "  ${CYAN}[SKIP]${NC} gRPC: ${name} ${DIM}- grpcurl not found${NC}"
    BASELINE_ENTRIES+=("{\"name\":\"gRPC: ${name//\"/\\\"}\",\"status\":0,\"result\":\"skip\"}")
    return
  fi

  local output
  output=$("$GRPCURL" \
    -import-path "$PROTO_DIR" \
    -proto api.proto \
    -H "authorization: "**********"
    -d "$json_data" \
    "$GRPC_HOST" \
    "$method" 2>&1) || true

  local has_error=false
  if echo "$output" | grep -qi "Code:\|rpc error" 2>/dev/null; then
    has_error=true
  fi

  if [[ "$expect_fail" == "true" ]]; then
    if [[ "$has_error" == true ]]; then
      EXPECTED_FAILED=$((EXPECTED_FAILED + 1))
      echo -e "  ${YELLOW}[EXPECTED_FAIL]${NC} gRPC: ${name} ${DIM}- (${fail_reason})${NC}"
      BASELINE_ENTRIES+=("{\"name\":\"gRPC: ${name//\"/\\\"}\",\"status\":0,\"result\":\"expected_fail\"}")
    else
      UNEXPECTED_PASS=$((UNEXPECTED_PASS + 1))
      PASSED=$((PASSED + 1))
      echo -e "  ${MAGENTA}[UNEXPECTED_PASS]${NC} gRPC: ${name} ${DIM}- Got success! (was: ${fail_reason})${NC}"
      UNEXPECTED_DETAILS+=("gRPC ${name}: Now succeeds (was: ${fail_reason})")
      BASELINE_ENTRIES+=("{\"name\":\"gRPC: ${name//\"/\\\"}\",\"status\":0,\"result\":\"unexpected_pass\"}")
    fi
  else
    if [[ "$has_error" == false ]]; then
      PASSED=$((PASSED + 1))
      echo -e "  ${GREEN}[PASS]${NC} gRPC: ${name} ${DIM}(${method})${NC}"
      if [[ "$VERBOSE" == true ]]; then
        echo -e "         ${DIM}${output:0:200}${NC}"
      fi
      BASELINE_ENTRIES+=("{\"name\":\"gRPC: ${name//\"/\\\"}\",\"status\":0,\"result\":\"pass\"}")
    else
      FAILED=$((FAILED + 1))
      echo -e "  ${RED}[FAIL]${NC} gRPC: ${name} ${DIM}(${method})${NC}"
      echo -e "         ${DIM}${output:0:200}${NC}"
      FAIL_DETAILS+=("gRPC ${name}: ${output:0:200}")
      BASELINE_ENTRIES+=("{\"name\":\"gRPC: ${name//\"/\\\"}\",\"status\":0,\"result\":\"fail\"}")
    fi
  fi
}

# ── section / should_run ──────────────────────────────────────────────────────
section() {
  echo ""
  echo -e "${BOLD}── $1 ──────────────────────────────────────────────${NC}"
}

should_run() {
  [[ -z "$SECTION_FILTER" ]] && return 0
  echo "$1" | grep -qi "$SECTION_FILTER" && return 0
  return 1
}


# ═══════════════════════════════════════════════════════════════════════════════
#  1. HEALTH
# ═══════════════════════════════════════════════════════════════════════════════
if should_run "health"; then
  section "1. Health"
  test_http "Health Check" "GET" "/"
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  2. SWAP v1
# ═══════════════════════════════════════════════════════════════════════════════
if should_run "swap"; then
  section "2. Swap v1"

  # GET /rpc/v1/swap/assets — asset cache must be warmed
  expected_fail \
    "Get Assets v1 (Base)" \
    "asset cache not warmed on dev" \
    "GET" "/rpc/v1/swap/assets?chainId=8453"

  # GET /rpc/v1/swap/asset — single asset lookup
  VALIDATE_FIELDS=("currencyCode" "name" "decimals" "chainId")
  test_http "Get Single Asset (USDC on Base)" \
    "GET" "/rpc/v1/swap/asset?chainId=8453&address=${BASE_USDC}"

  VALIDATE_FIELDS=("currencyCode" "name" "decimals")
  test_http "Get Single Asset (ETH native on Base)" \
    "GET" "/rpc/v1/swap/asset?chainId=8453&address=${ETH_NATIVE}"

  # GET /rpc/v1/swap/quote — various pairs
  VALIDATE_FIELDS=("fromAsset" "toAsset" "fromAmount")
  test_http "Quote v1 (ETH→USDC, Base)" \
    "GET" "/rpc/v1/swap/quote?chainId=8453&from=${ETH_NATIVE}&to=${BASE_USDC}&amount=${SMALL_ETH}&amountReference=from"

  VALIDATE_FIELDS=("fromAsset" "toAsset" "fromAmount")
  test_http "Quote v1 (USDC→ETH, Base)" \
    "GET" "/rpc/v1/swap/quote?chainId=8453&from=${BASE_USDC}&to=${ETH_NATIVE}&amount=${SMALL_USDC}&amountReference=from"

  VALIDATE_FIELDS=("fromAsset" "toAsset")
  test_http "Quote v1 (ETH→USDC, Mainnet)" \
    "GET" "/rpc/v1/swap/quote?chainId=1&from=${ETH_NATIVE}&to=${ETH_USDC}&amount=${SMALL_ETH}&amountReference=from"

# ═══════════════════════════════════════════════════════════════════════════════
#  3. SWAP v2
# ═══════════════════════════════════════════════════════════════════════════════
  section "3. Swap v2"

  # POST /rpc/v2/swap/assetsByContractAddress — batch asset lookup (chainId MUST be int)
  test_http "Assets by Contract Address (single)" \
    "POST" "/rpc/v2/swap/assetsByContractAddress" \
    "{\"8453\":[\"${BASE_USDC}\"]}"

  test_http "Assets by Contract Address (multiple)" \
    "POST" "/rpc/v2/swap/assetsByContractAddress" \
    "{\"8453\":[\"${BASE_USDC}\",\"${BASE_WETH}\"]}"

  test_http "Assets by Contract Address (ETH mainnet)" \
    "POST" "/rpc/v2/swap/assetsByContractAddress" \
    "{\"1\":[\"${ETH_USDC}\"]}"

  # GET /rpc/v2/swap/trade — EOA gets gas estimation
  VALIDATE_FIELDS=("fromAsset" "toAsset")
  test_http "Trade v2 (EOA, ETH→USDC, Base)" \
    "GET" "/rpc/v2/swap/trade?chainId=8453&from=${ETH_NATIVE}&to=${BASE_USDC}&amount=${SMALL_ETH}&amountReference=from&fromAddress=${EOA_WALLET}"

  # GET /rpc/v2/swap/trade — SCW skips gas estimation
  VALIDATE_FIELDS=("fromAsset" "toAsset")
  test_http "Trade v2 (SCW, ETH→USDC, Base)" \
    "GET" "/rpc/v2/swap/trade?chainId=8453&from=${ETH_NATIVE}&to=${BASE_USDC}&amount=${SMALL_ETH}&amountReference=from&fromAddress=${SCW_WALLET}" \
    "" "200" \
    "X-Wallet-Account-Type: 6"

  # Reverse direction
  VALIDATE_FIELDS=("fromAsset" "toAsset")
  test_http "Trade v2 (EOA, USDC→ETH, Base)" \
    "GET" "/rpc/v2/swap/trade?chainId=8453&from=${BASE_USDC}&to=${ETH_NATIVE}&amount=${SMALL_USDC}&amountReference=from&fromAddress=${EOA_WALLET}"

  # ETH mainnet
  VALIDATE_FIELDS=("fromAsset" "toAsset")
  test_http "Trade v2 (EOA, ETH→USDC, Mainnet)" \
    "GET" "/rpc/v2/swap/trade?chainId=1&from=${ETH_NATIVE}&to=${ETH_USDC}&amount=${SMALL_ETH}&amountReference=from&fromAddress=${EOA_WALLET}"

  # GET /rpc/v2/swap/supportedChains
  VALIDATE_FIELDS=("chainId")
  test_http "Supported Chains (EOA)" \
    "GET" "/rpc/v2/swap/supportedChains"

  VALIDATE_FIELDS=("chainId")
  test_http "Supported Chains (SCW)" \
    "GET" "/rpc/v2/swap/supportedChains" \
    "" "200" \
    "X-Wallet-Account-Type: 6"

  # POST /rpc/v2/swap/getRibbon
  expected_fail "Get Ribbon (Base)" "ribbon cache not warmed on dev" "POST" "/rpc/v2/swap/getRibbon" "{\"network\":\"networks/base-mainnet\",\"walletAddresses\":[\"${EOA_WALLET}\"]}"

# ═══════════════════════════════════════════════════════════════════════════════
#  4. SWAP v3
# ═══════════════════════════════════════════════════════════════════════════════
  section "4. Swap v3"

  VALIDATE_FIELDS=("fromAsset" "toAsset")
  test_http "Trade v3 (EOA, ETH→USDC, Base)" \
    "GET" "/rpc/v3/swap/trade?chainId=8453&from=${ETH_NATIVE}&to=${BASE_USDC}&amount=${SMALL_ETH}&amountReference=from&fromAddress=${EOA_WALLET}"

  VALIDATE_FIELDS=("fromAsset" "toAsset")
  test_http "Trade v3 (SCW, ETH→USDC, Base)" \
    "GET" "/rpc/v3/swap/trade?chainId=8453&from=${ETH_NATIVE}&to=${BASE_USDC}&amount=${SMALL_ETH}&amountReference=from&fromAddress=${SCW_WALLET}" \
    "" "200" \
    "X-Wallet-Account-Type: 6"

  VALIDATE_FIELDS=("fromAsset" "toAsset")
  test_http "Trade v3 (EOA, USDC→ETH, Base)" \
    "GET" "/rpc/v3/swap/trade?chainId=8453&from=${BASE_USDC}&to=${ETH_NATIVE}&amount=${SMALL_USDC}&amountReference=from&fromAddress=${EOA_WALLET}"

  VALIDATE_FIELDS=("fromAsset" "toAsset")
  test_http "Trade v3 (EOA, ETH→USDC, Mainnet)" \
    "GET" "/rpc/v3/swap/trade?chainId=1&from=${ETH_NATIVE}&to=${ETH_USDC}&amount=${SMALL_ETH}&amountReference=from&fromAddress=${EOA_WALLET}"
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  5. CROSS-CHAIN
# ═══════════════════════════════════════════════════════════════════════════════
if should_run "crosschain"; then
  section "5. Cross-Chain"

  # GET /rpc/v3/swap/crosschain/assets
  test_http "Cross-Chain Assets (Base→ETH)" \
    "GET" "/rpc/v3/swap/crosschain/assets?fromNetwork=networks/base-mainnet&toNetwork=networks/ethereum-mainnet"

  # Base USDC → ETH mainnet ETH
  VALIDATE_FIELDS=("fromAsset" "toAsset")
  test_http "Cross-Chain Quote (Base USDC→ETH mainnet)" \
    "GET" "/rpc/v3/swap/crosschain/quote?fromNetwork=networks/base-mainnet&toNetwork=networks/ethereum-mainnet&fromAssetAddress=${BASE_USDC}&toAssetAddress=${ETH_NATIVE}&fromAmount=${SMALL_USDC}&userAddress=${EOA_WALLET}"

  expected_fail "Cross-Chain Trade (EOA, Base USDC→ETH mainnet)" "amount parsing bug on dev backend" \
    "GET" "/rpc/v3/swap/crosschain/trade?fromNetwork=networks/base-mainnet&toNetwork=networks/ethereum-mainnet&fromAssetAddress=${BASE_USDC}&toAssetAddress=${ETH_NATIVE}&fromAmount=${SMALL_USDC}&userAddress=${EOA_WALLET}"

  expected_fail "Cross-Chain Trade (SCW, Base USDC→ETH mainnet)" "amount parsing bug on dev backend" \
    "GET" "/rpc/v3/swap/crosschain/trade?fromNetwork=networks/base-mainnet&toNetwork=networks/ethereum-mainnet&fromAssetAddress=${BASE_USDC}&toAssetAddress=${ETH_NATIVE}&fromAmount=${SMALL_USDC}&userAddress=${SCW_WALLET}" \
    "" "X-Wallet-Account-Type: 6"

  # Reverse: ETH mainnet → Base
  VALIDATE_FIELDS=("fromAsset" "toAsset")
  test_http "Cross-Chain Quote (ETH mainnet→Base USDC)" \
    "GET" "/rpc/v3/swap/crosschain/quote?fromNetwork=networks/ethereum-mainnet&toNetwork=networks/base-mainnet&fromAssetAddress=${ETH_NATIVE}&toAssetAddress=${BASE_USDC}&fromAmount=${SMALL_ETH}&userAddress=${EOA_WALLET}"

  expected_fail "Cross-Chain Trade (EOA, ETH mainnet→Base USDC)" "amount parsing bug on dev backend" \
    "GET" "/rpc/v3/swap/crosschain/trade?fromNetwork=networks/ethereum-mainnet&toNetwork=networks/base-mainnet&fromAssetAddress=${ETH_NATIVE}&toAssetAddress=${BASE_USDC}&fromAmount=${SMALL_ETH}&userAddress=${EOA_WALLET}"
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  6. GASLESS / META SWAP
# ═══════════════════════════════════════════════════════════════════════════════
if should_run "gasless"; then
  section "6. Gasless / Meta Swap"

  # GET /rpc/v1/gasless/swap/quote — EOA only
  expected_fail "Meta Quote (Base, USDC→WETH)" "gasless liquidity too low on dev" "GET" "/rpc/v1/gasless/swap/quote?chainId=8453&from=${BASE_USDC}&to=${BASE_WETH}&amount=${SMALL_USDC}&amountReference=from&fromAddress=${EOA_WALLET}"

  # GET /rpc/v1/gasless/swap/trade
  expected_fail "Meta Trade (Base, USDC→WETH, EOA)" "gasless metatrade not supported on dev" "GET" "/rpc/v1/gasless/swap/trade?chainId=8453&from=${BASE_USDC}&to=${BASE_WETH}&amount=${SMALL_USDC}&amountReference=from&fromAddress=${EOA_WALLET}"

  # POST /rpc/v1/gasless/swap/submit — requires signed tx
  skip "Meta Submit" "requires signed transaction data"

  # GET /rpc/v1/gasless/swap/status
  test_http "Meta Trade Status (dummy hash)" \
    "GET" "/rpc/v1/gasless/swap/status?tradeHash=0x0000000000000000000000000000000000000000000000000000000000000001&chainId=8453&aggregatorId=test"

  # GET /rpc/v1/gasless/swap/showGasless
  test_http "Show Gasless (Base, USDC→ETH)" \
    "GET" "/rpc/v1/gasless/swap/showGasless?chainId=8453&from=${BASE_USDC}&to=${ETH_NATIVE}&amount=${SMALL_USDC}&fromAddress=${EOA_WALLET}"

  test_http "Show Gasless (Base, ETH→USDC)" \
    "GET" "/rpc/v1/gasless/swap/showGasless?chainId=8453&from=${ETH_NATIVE}&to=${BASE_USDC}&amount=${SMALL_ETH}&fromAddress=${EOA_WALLET}"
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  7. BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════
if should_run "bridge"; then
  section "7. Bridge"

  # GET /rpc/v2/bridge/supportedChains
  test_http "Bridge Supported Chains" \
    "GET" "/rpc/v2/bridge/supportedChains"

  test_http "Bridge Supported Chains (filtered)" \
    "GET" "/rpc/v2/bridge/supportedChains?sourceChainId=8453"

  # GET /rpc/v2/bridge/supportedSourceAssets
  test_http "Bridge Source Assets (Base)" \
    "GET" "/rpc/v2/bridge/supportedSourceAssets?fromChainId=8453&toChainId=1"

  test_http "Bridge Source Assets (ETH mainnet)" \
    "GET" "/rpc/v2/bridge/supportedSourceAssets?fromChainId=1&toChainId=8453"

  # GET /rpc/v2/bridge/supportedTargetAssets
  test_http "Bridge Target Assets (Base USDC→ETH)" \
    "GET" "/rpc/v2/bridge/supportedTargetAssets?fromChainId=8453&toChainId=1&fromContractAddress=${BASE_USDC}"

  # GET /rpc/v2/bridge/quote
  test_http "Bridge Quote (Base USDC→ETH USDC)" \
    "GET" "/rpc/v2/bridge/quote?fromChainId=8453&toChainId=1&fromContractAddress=${BASE_USDC}&toContractAddress=${ETH_USDC}&fromCurrencyCode=USDC&toCurrencyCode=USDC&amount=${SMALL_USDC}&userAddress=${EOA_WALLET}"

  # GET /rpc/v2/bridge/getTxData
  expected_fail "Bridge Tx Data (Base USDC→ETH USDC)" "requires routePath from quote response" \
    "GET" "/rpc/v2/bridge/getTxData?fromChainId= "**********"=1&fromTokenAddress=${BASE_USDC}&toTokenAddress=${ETH_USDC}&fromAmount=${SMALL_USDC}&toAmount=900000&userAddress=${EOA_WALLET}"
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  8. INTENT
# ═══════════════════════════════════════════════════════════════════════════════
if should_run "intent"; then
  section "8. Intent"

  expected_fail \
    "Intent Get Quote" \
    "intent routes not enabled on dev" \
    "POST" "/rpc/v2/intent/getQuote" \
    "{\"fromAssetAddress\":\"${ETH_NATIVE}\",\"toAssetAddress\":\"${BASE_USDC}\",\"fromAmount\":\"${SMALL_ETH}\",\"userAddress\":\"${EOA_WALLET}\"}"

  expected_fail \
    "Intent Submit Quote" \
    "intent routes not enabled on dev" \
    "POST" "/rpc/v2/intent/submitQuote" \
    "{\"quoteId\":\"test-quote-id\"}"
fi

# ═══════════════════════════════════════════════════════════════════════════════
#  9. EARN / STAKING
# ═══════════════════════════════════════════════════════════════════════════════
if should_run "earn"; then
  section "9. Earn / Staking"

  test_http "Earn Supported Source Assets" \
    "GET" "/rpc/v2/earn/supportedSourceAssets?type=liquid_staking"

  test_http "Earn Supported Target Assets" \
    "GET" "/rpc/v2/earn/supportedTargetAssets?chainId=1&currencyCode=ETH"

  VALIDATE_FIELDS=("rate")
  test_http "Earn APR (ETH→rETH, mainnet)" \
    "GET" "/rpc/v2/earn/apr?chainId=1&sourceAsset=${ETH_NATIVE}&targetAsset=${RETH}"

  # Earn trade with both wallet types
  test_http "Earn Trade (EOA, ETH→rETH)" \
    "GET" "/rpc/v2/earn/trade?chainId=1&from=${ETH_NATIVE}&to=${RETH}&amount=${SMALL_ETH}&fromAddress=${EOA_WALLET}"

  test_http "Earn Trade (SCW, ETH→rETH)" \
    "GET" "/rpc/v2/earn/trade?chainId=1&from=${ETH_NATIVE}&to=${RETH}&amount=${SMALL_ETH}&fromAddress=${SCW_WALLET}" \
    "" "200" \
    "X-Wallet-Account-Type: 6"

  expected_fail "Earn Deposit (EOA, ETH→rETH)" "routing bug on dev" "GET" "/rpc/v2/earn/deposit?chainId=1&from=${ETH_NATIVE}&to=${RETH}&amount=${SMALL_ETH}&userAddress=${EOA_WALLET}"

  expected_fail "Earn Withdraw (EOA, rETH→ETH)" "routing bug on dev" "GET" "/rpc/v2/earn/withdraw?chainId=1&from=${RETH}&to=${ETH_NATIVE}&amount=${SMALL_ETH}&userAddress=${EOA_WALLET}"

  expected_fail "Earn Claim (EOA, mainnet)" "tokenId param rejected on dev" "GET" "/rpc/v2/earn/claim?chainId= "**********"=${EOA_WALLET}"

  expected_fail "Earn Claim Status (EOA, mainnet)" "tokenId param rejected on dev" "GET" "/rpc/v2/earn/claimStatus?chainId= "**********"=${EOA_WALLET}"

  test_http "Staking Supported Assets" \
    "GET" "/rpc/v2/staking/supportedAssets"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 10. ETHEREUM KILN
# ═══════════════════════════════════════════════════════════════════════════════
if should_run "kiln"; then
  section "10. Ethereum Kiln"

  expected_fail "Kiln Stake (EOA)" "Kiln backend down on dev" "GET" "/rpc/v2/earn/ethereum_kiln/stake?amount=${SMALL_ETH}&userAddress=${EOA_WALLET}"

  expected_fail "Kiln Stake (SCW)" "Kiln backend down on dev" "GET" "/rpc/v2/earn/ethereum_kiln/stake?amount=${SMALL_ETH}&userAddress=${SCW_WALLET}"

  expected_fail "Kiln Unstake (EOA)" "Kiln goerli deprecated" "GET" "/rpc/v2/earn/ethereum_kiln/unstake?amount=${SMALL_ETH}&userAddress=${EOA_WALLET}"

  expected_fail "Kiln Claim (EOA)" "Kiln goerli deprecated" "GET" "/rpc/v2/earn/ethereum_kiln/claim?userAddress=${EOA_WALLET}"

  expected_fail "Kiln Submit (dummy tx)" "Kiln backend down on dev" "POST" "/rpc/v2/earn/ethereum_kiln/submit" "{\"workflowName\":\"test\",\"hash\":\"0x0000000000000000000000000000000000000000000000000000000000000001\"}"

  expected_fail \
    "Kiln Balance" \
    "Kiln dev backend pointed at goerli (deprecated)" \
    "GET" "/rpc/v2/earn/ethereum_kiln/balance?userAddress=${EOA_WALLET}"

  test_http "Kiln Networks" \
    "GET" "/rpc/v2/earn/ethereum_kiln/networks"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 11. PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
if should_run "portfolio"; then
  section "11. Portfolio"

  # GET /rpc/v2/portfolio/getPortfolio — uses ?id= NOT ?address=
  test_http "Get Portfolio (EOA)" \
    "GET" "/rpc/v2/portfolio/getPortfolio?id=${EOA_WALLET}&chainId=1"

  test_http "Get Portfolio (SCW)" \
    "GET" "/rpc/v2/portfolio/getPortfolio?id=${SCW_WALLET}&chainId=1"

  # POST uses evmWalletAddresses NOT addresses
  test_http "DeFi Protocol Portfolio (EOA)" \
    "POST" "/rpc/v2/portfolio/defi/protocol" \
    "{\"evmWalletAddresses\":[\"${EOA_WALLET}\"]}"

  test_http "DeFi Protocol Portfolio (SCW)" \
    "POST" "/rpc/v2/portfolio/defi/protocol" \
    "{\"evmWalletAddresses\":[\"${SCW_WALLET}\"]}"

  test_http "DeFi Earn Portfolio (EOA)" \
    "POST" "/rpc/v2/portfolio/defi/earn" \
    "{\"evmWalletAddresses\":[\"${EOA_WALLET}\"]}"

  test_http "DeFi Get Assets" \
    "GET" "/rpc/v2/portfolio/defi/get-assets"

  test_http "DeFi Claims (EOA)" \
    "POST" "/rpc/v2/portfolio/defi/claims" \
    "{\"evmWalletAddresses\":[\"${EOA_WALLET}\"]}"

  test_http "DeFi LP Tokens" \
    "GET" "/rpc/v2/portfolio/defi/lp-tokens"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 12. EXPLORE
# ═══════════════════════════════════════════════════════════════════════════════
if should_run "explore"; then
  section "12. Explore"

  test_http "Trending Swaps" \
    "GET" "/rpc/v2/explore/trendingSwaps"

  test_http "Trending Swaps by Network (Base)" \
    "GET" "/rpc/v2/explore/trendingSwapsByNetworkId?networkId=networks/base-mainnet"

  test_http "Trending Swaps by Network (ETH)" \
    "GET" "/rpc/v2/explore/trendingSwapsByNetworkId?networkId=networks/ethereum-mainnet"

  expected_fail \
    "Swap Volume by Asset (Base)" \
    "background job data not populated on dev" \
    "GET" "/rpc/v2/explore/swapVolumeByAsset?networkId=networks/base-mainnet"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 13. INTERNAL
# ═══════════════════════════════════════════════════════════════════════════════
if should_run "internal"; then
  section "13. Internal"

  expected_fail \
    "Internal Asset by Address" \
    "internal routes not exposed on dev" \
    "GET" "/internal/assets/byAddress?address=${BASE_USDC}&chainId=8453"

  expected_fail \
    "Internal Assets by Contracts Search" \
    "internal routes not exposed on dev" \
    "GET" "/internal/assets/byContractsSearch?search=USDC&chainId=8453"

  expected_fail \
    "Internal Set Jupiter Fee Account" \
    "internal routes not exposed on dev" \
    "POST" "/internal/setJupiterFeeAccount" \
    "{}"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 14. gRPC
# ═══════════════════════════════════════════════════════════════════════════════
if should_run "grpc"; then
  section "14. gRPC"

  # GetQuote — same-chain
  test_grpc "GetQuote (ETH→USDC, Base)" \
    "swapservice.SwapService/GetQuote" \
    "{\"from_asset_address\":\"${ETH_NATIVE}\",\"to_asset_address\":\"${BASE_USDC}\",\"from_network\":\"networks/base-mainnet\",\"to_network\":\"networks/base-mainnet\",\"from_amount\":\"${SMALL_ETH}\",\"user_address\":\"${EOA_WALLET}\"}"

  # GetQuote — cross-chain
  test_grpc "GetQuote (Base USDC→ETH mainnet)" \
    "swapservice.SwapService/GetQuote" \
    "{\"from_asset_address\":\"${BASE_USDC}\",\"to_asset_address\":\"${ETH_NATIVE}\",\"from_network\":\"networks/base-mainnet\",\"to_network\":\"networks/ethereum-mainnet\",\"from_amount\":\"${SMALL_USDC}\",\"user_address\":\"${EOA_WALLET}\"}"

  # GetTrade — same-chain EOA
  test_grpc "GetTrade (EOA, ETH→USDC, Base)" \
    "swapservice.SwapService/GetTrade" \
    "{\"from_asset_address\":\"${ETH_NATIVE}\",\"to_asset_address\":\"${BASE_USDC}\",\"from_network\":\"networks/base-mainnet\",\"to_network\":\"networks/base-mainnet\",\"from_amount\":\"${SMALL_ETH}\",\"user_address\":\"${EOA_WALLET}\"}"

  # GetTrade — cross-chain
  test_grpc "GetTrade (EOA, Base USDC→ETH mainnet)" \
    "swapservice.SwapService/GetTrade" \
    "{\"from_asset_address\":\"${BASE_USDC}\",\"to_asset_address\":\"${ETH_NATIVE}\",\"from_network\":\"networks/base-mainnet\",\"to_network\":\"networks/ethereum-mainnet\",\"from_amount\":\"${SMALL_USDC}\",\"user_address\":\"${EOA_WALLET}\"}"

  # GetAsset
  test_grpc "GetAsset (USDC on Base)" \
    "swapservice.SwapService/GetAsset" \
    "{\"chain_id\":\"8453\",\"contract_address\":\"${BASE_USDC}\"}"

  test_grpc "GetAsset (ETH native on mainnet)" \
    "swapservice.SwapService/GetAsset" \
    "{\"chain_id\":\"1\",\"contract_address\":\"${ETH_NATIVE}\"}"

  # GetAssets — needs asset cache
  test_grpc "GetAssets (Base→ETH)" \
    "swapservice.SwapService/GetAssets" \
    "{\"from_network\":\"networks/base-mainnet\",\"to_network\":\"networks/ethereum-mainnet\"}"

  # GetSupportedChains
  test_grpc "GetSupportedChains" \
    "swapservice.SwapService/GetSupportedChains" \
    "{}"

  # GetDefaultSwapAssets
  test_grpc "GetDefaultSwapAssets" \
    "swapservice.SwapService/GetDefaultSwapAssets" \
    "{}"

  # IngestAsset
  test_grpc "IngestAsset (USDC on Base)" \
    "swapservice.SwapService/IngestAsset" \
    "{\"chain_id\":\"8453\",\"contract_address\":\"${BASE_USDC}\"}"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# 15. WORKFLOW: Quote → Trade end-to-end
# ═══════════════════════════════════════════════════════════════════════════════
if should_run "workflow"; then
  section "15. Workflows (Quote → Trade)"

  echo -e "  ${DIM}── Same-chain EOA (Base ETH→USDC) ──${NC}"
  VALIDATE_FIELDS=("fromAsset" "toAsset" "fromAmount")
  test_http "  WF: Quote (EOA, ETH→USDC)" \
    "GET" "/rpc/v1/swap/quote?chainId=8453&from=${ETH_NATIVE}&to=${BASE_USDC}&amount=${SMALL_ETH}&amountReference=from"
  VALIDATE_FIELDS=("fromAsset" "toAsset")
  test_http "  WF: Trade v2 (EOA, ETH→USDC)" \
    "GET" "/rpc/v2/swap/trade?chainId=8453&from=${ETH_NATIVE}&to=${BASE_USDC}&amount=${SMALL_ETH}&amountReference=from&fromAddress=${EOA_WALLET}"

  echo -e "  ${DIM}── Same-chain SCW (Base ETH→USDC) ──${NC}"
  VALIDATE_FIELDS=("fromAsset" "toAsset")
  test_http "  WF: Quote (SCW, ETH→USDC)" \
    "GET" "/rpc/v1/swap/quote?chainId=8453&from=${ETH_NATIVE}&to=${BASE_USDC}&amount=${SMALL_ETH}&amountReference=from" \
    "" "200" \
    "X-Wallet-Account-Type: 6"
  VALIDATE_FIELDS=("fromAsset" "toAsset")
  test_http "  WF: Trade v2 (SCW, ETH→USDC)" \
    "GET" "/rpc/v2/swap/trade?chainId=8453&from=${ETH_NATIVE}&to=${BASE_USDC}&amount=${SMALL_ETH}&amountReference=from&fromAddress=${SCW_WALLET}" \
    "" "200" \
    "X-Wallet-Account-Type: 6"

  echo -e "  ${DIM}── Cross-chain EOA (Base USDC→ETH mainnet) ──${NC}"
  VALIDATE_FIELDS=("fromAsset" "toAsset")
  test_http "  WF: Cross-Chain Quote (EOA)" \
    "GET" "/rpc/v3/swap/crosschain/quote?fromNetwork=networks/base-mainnet&toNetwork=networks/ethereum-mainnet&fromAssetAddress=${BASE_USDC}&toAssetAddress=${ETH_NATIVE}&fromAmount=${SMALL_USDC}&userAddress=${EOA_WALLET}"
  expected_fail "  WF: Cross-Chain Trade (EOA)" "amount parsing bug on dev" \
    "GET" "/rpc/v3/swap/crosschain/trade?fromNetwork=networks/base-mainnet&toNetwork=networks/ethereum-mainnet&fromAssetAddress=${BASE_USDC}&toAssetAddress=${ETH_NATIVE}&fromAmount=${SMALL_USDC}&userAddress=${EOA_WALLET}"

  echo -e "  ${DIM}── gRPC EOA (Base ETH→USDC) ──${NC}"
  test_grpc "  WF: gRPC GetQuote (EOA)" \
    "swapservice.SwapService/GetQuote" \
    "{\"from_asset_address\":\"${ETH_NATIVE}\",\"to_asset_address\":\"${BASE_USDC}\",\"from_network\":\"networks/base-mainnet\",\"to_network\":\"networks/base-mainnet\",\"from_amount\":\"${SMALL_ETH}\",\"user_address\":\"${EOA_WALLET}\"}"
  test_grpc "  WF: gRPC GetTrade (EOA)" \
    "swapservice.SwapService/GetTrade" \
    "{\"from_asset_address\":\"${ETH_NATIVE}\",\"to_asset_address\":\"${BASE_USDC}\",\"from_network\":\"networks/base-mainnet\",\"to_network\":\"networks/base-mainnet\",\"from_amount\":\"${SMALL_ETH}\",\"user_address\":\"${EOA_WALLET}\"}"

  echo -e "  ${DIM}── Bridge EOA (Base USDC→ETH USDC) ──${NC}"
  test_http "  WF: Bridge Quote (EOA)" \
    "GET" "/rpc/v2/bridge/quote?fromChainId=8453&toChainId=1&fromContractAddress=${BASE_USDC}&toContractAddress=${ETH_USDC}&fromCurrencyCode=USDC&toCurrencyCode=USDC&amount=${SMALL_USDC}&userAddress=${EOA_WALLET}"
  expected_fail "  WF: Bridge Tx Data (EOA)" "requires routePath from quote" \
    "GET" "/rpc/v2/bridge/getTxData?fromChainId= "**********"=1&fromTokenAddress=${BASE_USDC}&toTokenAddress=${ETH_USDC}&fromAmount=${SMALL_USDC}&toAmount=900000&userAddress=${EOA_WALLET}"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}  SMOKE TEST SUMMARY${NC}"
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"
printf "  %-20s ${BOLD}%d${NC}\n" "Total:" "$TOTAL"
printf "  ${GREEN}%-20s %d${NC}\n" "Passed:" "$PASSED"
printf "  ${RED}%-20s %d${NC}\n" "Failed:" "$FAILED"
printf "  ${YELLOW}%-20s %d${NC}\n" "Expected Fail:" "$EXPECTED_FAILED"
printf "  ${CYAN}%-20s %d${NC}\n" "Skipped:" "$SKIPPED"
if [[ $UNEXPECTED_PASS -gt 0 ]]; then
  printf "  ${MAGENTA}%-20s %d  ← issues may be fixed!${NC}\n" "Unexpected Pass:" "$UNEXPECTED_PASS"
fi
echo -e "${BOLD}═══════════════════════════════════════════════════════════════${NC}"

# Failure details
if [[ ${#FAIL_DETAILS[@]} -gt 0 ]]; then
  echo ""
  echo -e "${RED}${BOLD}Failed Tests:${NC}"
  for detail in "${FAIL_DETAILS[@]}"; do
    echo -e "  ${RED}•${NC} ${detail}"
  done
fi

# Unexpected passes
if [[ ${#UNEXPECTED_DETAILS[@]} -gt 0 ]]; then
  echo ""
  echo -e "${MAGENTA}${BOLD}Unexpected Passes (good news — may be fixed!):${NC}"
  for detail in "${UNEXPECTED_DETAILS[@]}"; do
    echo -e "  ${MAGENTA}•${NC} ${detail}"
  done
fi

# ── Save baseline JSON ───────────────────────────────────────────────────────
if [[ "$SAVE_BASELINE" == true ]]; then
  BASELINE_FILE="${SCRIPT_DIR}/baseline.json"
  {
    echo "["
    for i in "${!BASELINE_ENTRIES[@]}"; do
      if [[ $i -gt 0 ]]; then
        echo ","
      fi
      echo -n "  ${BASELINE_ENTRIES[$i]}"
    done
    echo ""
    echo "]"
  } > "$BASELINE_FILE"
  echo ""
  echo -e "${DIM}Baseline saved to ${BASELINE_FILE}${NC}"
fi

echo ""

if [[ $FAILED -gt 0 ]]; then
  exit 1
fi
exit 0
