#date: 2026-02-09T17:27:28Z
#url: https://api.github.com/gists/fcc64e118de5a556ab57c8591bd3feb7
#owner: https://api.github.com/users/sudogeeker

#!/usr/bin/env bash
set -euo pipefail

# ---------------- UI ----------------
if command -v tput >/dev/null 2>&1; then
  B="$(tput bold)"; D="$(tput dim)"; R="$(tput sgr0)"
  RED="$(tput setaf 1)"; GRN="$(tput setaf 2)"; YLW="$(tput setaf 3)"; BLU="$(tput setaf 4)"; CYN="$(tput setaf 6)"
else
  B=""; D=""; R=""; RED=""; GRN=""; YLW=""; BLU=""; CYN=""
fi
hr(){ printf "%s\n" "${D}────────────────────────────────────────────────────────${R}"; }
ok(){ printf "%s[OK]%s %s\n" "$GRN" "$R" "$*"; }
info(){ printf "%s[INFO]%s %s\n" "$BLU" "$R" "$*"; }
warn(){ printf "%s[WARN]%s %s\n" "$YLW" "$R" "$*"; }
die(){ printf "%s[ERR]%s %s\n" "$RED" "$R" "$*" >&2; exit 1; }

ask() {
  local prompt="$1" default="${2-}"
  if [[ -n "${default}" ]]; then
    printf "%s?%s %s %s[%s]%s: " "$CYN" "$R" "$prompt" "$D" "$default" "$R" >&2
  else
    printf "%s?%s %s: " "$CYN" "$R" "$prompt" >&2
  fi
  local ans; IFS= read -r ans
  [[ -z "$ans" && -n "${default}" ]] && ans="$default"
  printf "%s" "$ans"
}

require(){ command -v "$1" >/dev/null 2>&1 || die "缺少命令：$1"; }
is_root(){ [[ "$(id -u)" -eq 0 ]]; }

# ---------------- args ----------------
USE_PMTU=0
BASE_MTU_OVERRIDE=""
OUTDIR="/etc/network/interfaces.d"

usage(){
  cat <<EOF
用法:
  $0 [--pmtu] [--mtu N] [--outdir DIR]

选项:
  --pmtu        用 tracepath 探测到对端的 PMTU 作为基准 MTU
  --mtu N       手动指定基准 MTU（同时给 --pmtu 时，以 --pmtu 为准）
  --outdir DIR  输出目录（默认 /etc/network/interfaces.d）
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pmtu) USE_PMTU=1; shift ;;
    --mtu) BASE_MTU_OVERRIDE="${2-}"; shift 2 ;;
    --outdir) OUTDIR="${2-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "未知参数：$1（用 -h 查看帮助）" ;;
  esac
done

# ---------------- checks ----------------
is_root || die "请用 root 执行（sudo -i）"
require ip
require openssl

# ---------------- helpers ----------------
default_dev() {
  local fam="$1"
  if [[ "$fam" == "4" ]]; then
    ip -4 route show default 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}'
  else
    ip -6 route show default 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}'
  fi
}
dev_mtu() {
  local dev="$1"
  ip link show dev "$dev" | awk '{for(i=1;i<=NF;i++) if($i=="mtu"){print $(i+1); exit}}'
}
route_src_dev() {
  local fam="$1" remote="$2"
  if [[ "$fam" == "4" ]]; then
    ip -4 route get "$remote" 2>/dev/null | awk '{for(i=1;i<=NF;i++){if($i=="dev"){d=$(i+1)} if($i=="src"){s=$(i+1)}}} END{print d" "s}'
  else
    ip -6 route get "$remote" 2>/dev/null | awk '{for(i=1;i<=NF;i++){if($i=="dev"){d=$(i+1)} if($i=="src"){s=$(i+1)}}} END{print d" "s}'
  fi
}
detect_pmtu() {
  local fam="$1" remote="$2"
  command -v tracepath >/dev/null 2>&1 || return 1
  local out
  if [[ "$fam" == "4" ]]; then
    out="$(tracepath -n -m 6 -q 1 "$remote" 2>/dev/null || true)"
  else
    out="$(tracepath -6 -n -m 6 -q 1 "$remote" 2>/dev/null || true)"
  fi
  local pmtu
  pmtu="$(printf "%s\n" "$out" | awk '/pmtu/ {for(i=1;i<=NF;i++) if($i=="pmtu"){print $(i+1); exit}}')"
  [[ -n "$pmtu" ]] || return 1
  printf "%s" "$pmtu"
}
gen_keymat_20B_hex(){ openssl rand -hex 20; } # 20B = 40 hex (rfc4106 needs 16B key + 4B salt)
sha32() { # first 8 hex of sha256(input)
  printf "%s" "$1" | openssl dgst -sha256 -r | awk '{print $1}' | cut -c1-8
}
mk_spi_pair() {
  # deterministic across both ends: sort two underlay IP strings
  local name="$1" ip1="$2" ip2="$3"
  local a b
  a="$(printf "%s\n%s\n" "$ip1" "$ip2" | sort | sed -n '1p')"
  b="$(printf "%s\n%s\n" "$ip1" "$ip2" | sort | sed -n '2p')"
  local h1 h2
  h1="$(sha32 "xfrm-gre|$name|$a|$b|AB")"
  h2="$(sha32 "xfrm-gre|$name|$a|$b|BA")"
  # avoid 0
  [[ "$h1" == "00000000" ]] && h1="00003001"
  [[ "$h2" == "00000000" ]] && h2="00003002"
  printf "0x%s 0x%s %s %s\n" "$h1" "$h2" "$a" "$b"
}

# overhead model (conservative): ESP(GCM) ~ 32B (ESP hdr+IV+ICV), GRE 4B, outer IP hdr 20/40
ESP_OVH=32
GRE_OVH=4
IP4_HDR=20
IP6_HDR=40

# ---------------- interactive ----------------
hr
printf "%s%sGRE(L3) + XFRM(ESP) interfaces 生成器%s\n" "$B" "$CYN" "$R"
printf "%s只加密 GRE(proto47)，不劫持其它流量；PSK 只问一次%s\n" "$D" "$R"
hr

ufam="$(ask "选择 underlay IP 版本 (4/6)" "6")"
[[ "$ufam" == "4" || "$ufam" == "6" ]] || die "只能选 4 或 6"

ddev="$(default_dev "$ufam")"
dev="$(ask "主网卡(用于默认 src/MTU)" "${ddev:-eth0}")"

remote_underlay="$(ask "对端 underlay IP" "")"
[[ -n "$remote_underlay" ]] || die "对端 underlay IP 不能为空"

read -r route_dev route_src < <(route_src_dev "$ufam" "$remote_underlay")
[[ -n "${route_dev:-}" ]] || route_dev="$dev"
[[ -n "${route_src:-}" ]] || warn "未能自动得到 src 地址，你需要手填本机 underlay IP"

local_underlay="$(ask "本机 underlay IP" "${route_src:-}")"
[[ -n "$local_underlay" ]] || die "本机 underlay IP 不能为空"

name="$(ask "隧道名字（最终接口名：gre-<name>）" "prod1")"
[[ "$name" =~ ^[a-zA-Z0-9._-]+$ ]] || die "名字只能包含 a-zA-Z0-9 . _ -"
gre_if="gre-$name"

innerfam="$(ask "GRE 内层 IP 版本 (4/6)" "4")"
[[ "$innerfam" == "4" || "$innerfam" == "6" ]] || die "只能选 4 或 6"

if [[ "$innerfam" == "4" ]]; then
  inner_local_cidr="$(ask "本端 GRE 内层地址/CIDR（用于测试/业务，可后改）" "10.255.255.1/30")"
else
  inner_local_cidr="$(ask "本端 GRE 内层地址/CIDR（用于测试/业务，可后改）" "fd00:255::1/127")"
fi

psk="$(ask "XFRM PSK（40个hex；留空自动生成）" "")"
if [[ -z "$psk" ]]; then
  psk="$(gen_keymat_20B_hex)"
  ok "已生成 PSK(keymat 20B/40hex)：$psk"
else
  [[ "$psk" =~ ^[0-9a-fA-F]{40}$ ]] || die "PSK 必须是 40 个 hex（20字节，rfc4106 keymat）"
fi
psk="0x${psk,,}" # lower + 0x

# base MTU selection
base_mtu=""
if [[ "$USE_PMTU" -eq 1 ]]; then
  info "正在探测 PMTU（tracepath）…"
  if p="$(detect_pmtu "$ufam" "$remote_underlay")"; then
    base_mtu="$p"; ok "PMTU=$base_mtu"
  else
    warn "PMTU 探测失败，将回退到 --mtu 或 出口网卡 MTU"
  fi
fi
if [[ -z "$base_mtu" && -n "$BASE_MTU_OVERRIDE" ]]; then
  base_mtu="$BASE_MTU_OVERRIDE"; ok "使用手动基准 MTU：$base_mtu"
fi
if [[ -z "$base_mtu" ]]; then
  m="$(dev_mtu "$route_dev" 2>/dev/null || true)"
  [[ -n "$m" ]] || m="$(dev_mtu "$dev" 2>/dev/null || true)"
  [[ -n "$m" ]] || m="1500"
  base_mtu="$m"
  ok "使用出口网卡 $route_dev 的 MTU 作为基准：$base_mtu"
fi
[[ "$base_mtu" =~ ^[0-9]+$ ]] || die "基准 MTU 必须是数字"

# compute GRE MTU (conservative, leave room for ESP+GRE+outer IP hdr)
if [[ "$ufam" == "4" ]]; then
  gre_mtu=$(( base_mtu - IP4_HDR - GRE_OVH - ESP_OVH ))
  gre_type="gre"
  gre_mod="ip_gre"
  gre_args="local $local_underlay remote $remote_underlay ttl 64"
else
  gre_mtu=$(( base_mtu - IP6_HDR - GRE_OVH - ESP_OVH ))
  gre_type="ip6gre"
  gre_mod="ip6_gre"
  gre_args="local $local_underlay remote $remote_underlay hoplimit 64"
fi
(( gre_mtu >= 1280 )) || warn "计算得到 GRE MTU=$gre_mtu（偏小）。如有需要可用 --mtu 或调大基准"

# SPIs (deterministic)
read -r spi_ab spi_ba ipA ipB < <(mk_spi_pair "$name" "$local_underlay" "$remote_underlay")
# define who is A-side
if [[ "$local_underlay" == "$ipA" ]]; then
  spi_out="$spi_ab"   # local(A) -> remote(B)
  spi_in="$spi_ba"    # remote(B) -> local(A)
else
  spi_out="$spi_ba"   # local(B) -> remote(A)
  spi_in="$spi_ab"
fi

hr
info "underlay IPv$ufam: local=$local_underlay  remote=$remote_underlay  (route dev=$route_dev)"
info "GRE iface: ${B}$gre_if${R}  type=$gre_type  mtu=$gre_mtu  inner=$inner_local_cidr"
info "XFRM: algo=rfc4106(gcm(aes)) icvlen=128  spi_out=$spi_out spi_in=$spi_in"
hr

mkdir -p "$OUTDIR"
outfile="$OUTDIR/${gre_if}.cfg"

# ensure /etc/network/interfaces sources interfaces.d
if [[ -f /etc/network/interfaces ]]; then
  if ! grep -Eq '^\s*source\s+/etc/network/interfaces\.d/\*' /etc/network/interfaces; then
    warn "/etc/network/interfaces 未发现：source /etc/network/interfaces.d/*"
    ans="$(ask "是否自动追加该行？(y/N)" "N")"
    if [[ "$ans" =~ ^[yY]$ ]]; then
      printf "\nsource /etc/network/interfaces.d/*\n" >> /etc/network/interfaces
      ok "已追加 source 行到 /etc/network/interfaces"
    else
      warn "未追加。请自行确保 ifupdown 会加载 interfaces.d"
    fi
  fi
fi

tmp="$(mktemp)"

cat >"$tmp" <<EOF
# Generated by mk-gre-xfrm.sh
# underlay IPv$ufam: $local_underlay <-> $remote_underlay
# GRE: $gre_if ($gre_type) mtu=$gre_mtu inner=$inner_local_cidr
# XFRM: ESP transport (proto50) protects GRE(proto47) only
# PSK keymat (rfc4106) + ICV 128
# NOTE: contains PSK, keep file permission strict (600)

auto $gre_if
iface $gre_if inet manual
    pre-up  modprobe $gre_mod || :
    pre-up  ip link replace $gre_if type $gre_type $gre_args
    pre-up  ip link set dev $gre_if mtu $gre_mtu

    # --- XFRM: idempotent cleanup (only our entries) ---
    pre-up  /bin/sh -c 'set +e; \
            ip xfrm policy del dir out src $local_underlay/$( [[ "$ufam" == "4" ]] && echo 32 || echo 128 ) dst $remote_underlay/$( [[ "$ufam" == "4" ]] && echo 32 || echo 128 ) proto gre >/dev/null 2>&1; \
            ip xfrm policy del dir in  src $remote_underlay/$( [[ "$ufam" == "4" ]] && echo 32 || echo 128 ) dst $local_underlay/$( [[ "$ufam" == "4" ]] && echo 32 || echo 128 ) proto gre >/dev/null 2>&1; \
            ip xfrm state  del src $local_underlay dst $remote_underlay proto esp spi $spi_out >/dev/null 2>&1; \
            ip xfrm state  del src $remote_underlay dst $local_underlay proto esp spi $spi_in  >/dev/null 2>&1; \
            exit 0'

    # --- XFRM states (both directions) ---
    pre-up  ip xfrm state add src $local_underlay dst $remote_underlay proto esp spi $spi_out mode transport \\
            aead 'rfc4106(gcm(aes))' $psk 128
    pre-up  ip xfrm state add src $remote_underlay dst $local_underlay proto esp spi $spi_in  mode transport \\
            aead 'rfc4106(gcm(aes))' $psk 128

    # --- XFRM policy: protect GRE only (safe) ---
    pre-up  ip xfrm policy add dir out src $local_underlay/$( [[ "$ufam" == "4" ]] && echo 32 || echo 128 ) dst $remote_underlay/$( [[ "$ufam" == "4" ]] && echo 32 || echo 128 ) proto gre \\
            tmpl src $local_underlay dst $remote_underlay proto esp mode transport
    pre-up  ip xfrm policy add dir in  src $remote_underlay/$( [[ "$ufam" == "4" ]] && echo 32 || echo 128 ) dst $local_underlay/$( [[ "$ufam" == "4" ]] && echo 32 || echo 128 ) proto gre \\
            tmpl src $remote_underlay dst $local_underlay proto esp mode transport

    up      ip link set dev $gre_if up
EOF

# add inner addr setup
if [[ "$innerfam" == "4" ]]; then
  cat >>"$tmp" <<EOF
    post-up ip addr add $inner_local_cidr dev $gre_if 2>/dev/null || true
    pre-down ip addr del $inner_local_cidr dev $gre_if 2>/dev/null || true
EOF
else
  cat >>"$tmp" <<EOF
    post-up ip -6 addr add $inner_local_cidr dev $gre_if 2>/dev/null || true
    pre-down ip -6 addr del $inner_local_cidr dev $gre_if 2>/dev/null || true
EOF
fi

cat >>"$tmp" <<EOF
    down    ip link set dev $gre_if down
    post-down ip link del dev $gre_if
EOF

install -m 600 "$tmp" "$outfile"
rm -f "$tmp"
ok "已写入并设置权限 600：$outfile"

hr
printf "%s下一步在本机执行：%s\n" "$B" "$R"
cat <<EOF
  ifdown --force $gre_if 2>/dev/null || true
  ifup $gre_if

  # 看命中计数（必须增长）
  ip -s xfrm policy
  ip -s xfrm state

  # 抓包：应该看到 ESP(proto50)，明文 GRE(proto47) 应减少
  tcpdump -nni $route_dev 'ip${ufam} proto 50'
EOF
hr
printf "%s对端需要填同样的：name、两端 underlay IP、同一 PSK。%s\n" "$D" "$R"
printf "%s本次 PSK（请复制给对端）：%s%s%s\n" "$YLW" "$B" "${psk#0x}" "$R"
hr
