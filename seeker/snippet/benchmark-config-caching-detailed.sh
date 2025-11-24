#date: 2025-11-24T17:06:33Z
#url: https://api.github.com/gists/1cba0123c12a1bae96f54f1e2363af49
#owner: https://api.github.com/users/kibotu

#!/bin/bash

# Enhanced Configuration Cache Benchmark with Diagnostics
# This version provides detailed insights into cache behavior

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
RUNS=5
WARMUP_RUNS=2

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Enhanced Configuration Cache Benchmark              ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}\n"

# Arrays to store durations and cache stats
declare -a durations_no_cache
declare -a durations_with_cache
declare -a durations_no_cache_clean
declare -a durations_with_cache_clean
declare -a cache_hits
declare -a cache_misses

# Function to check cache status
check_cache_status() {
    local log_file=$1
    # Check for various cache reuse indicators
    if grep -q "Reusing configuration cache" "$log_file" 2>/dev/null || \
       grep -q "Configuration cache entry reused" "$log_file" 2>/dev/null || \
       grep -q "Reusing configuration from cache" "$log_file" 2>/dev/null; then
        echo "HIT"
    # Check for cache calculation/miss indicators
    elif grep -q "Calculating task graph" "$log_file" 2>/dev/null || \
         grep -q "Configuration cache entry stored" "$log_file" 2>/dev/null || \
         grep -q "Storing configuration cache" "$log_file" 2>/dev/null; then
        echo "MISS"
    else
        echo "UNKNOWN"
    fi
}

# Function to extract configuration time
extract_config_time() {
    local log_file=$1
    # Try to extract configuration time from build scan or logs
    local config_time=$(grep -i "configuration time" "$log_file" 2>/dev/null | grep -oE '[0-9]+(\.[0-9]+)?s' | head -1 | sed 's/s//')
    echo "${config_time:-0}"
}

# Test scenarios
echo -e "${CYAN}Test Configuration:${NC}"
echo -e "  • Runs per scenario: ${RUNS} (+ ${WARMUP_RUNS} warmup)"
echo -e "  • Scenarios: Clean build, Incremental build (no changes)"
echo -e "  • Daemon: Enabled with fresh start per scenario"
echo -e "  • Network: Offline mode (no dependency downloads)"
echo -e "  • Diagnostics: Cache hit/miss tracking enabled\n"

# Ensure dependencies are downloaded first
echo -e "${MAGENTA}Ensuring dependencies are cached...${NC}"
./gradlew build -x test --no-daemon > /dev/null 2>&1
echo -e "${GREEN}✓ Dependencies cached${NC}"

# Kill any existing daemons
echo -e "${MAGENTA}Stopping any existing Gradle daemons...${NC}"
./gradlew --stop > /dev/null 2>&1
sleep 2
echo -e "${GREEN}✓ Daemons stopped${NC}"

# Clean up old configuration cache
echo -e "${MAGENTA}Cleaning old configuration cache...${NC}"
rm -rf .gradle
echo -e "${GREEN}✓ Configuration cache cleaned${NC}"

# Warmup daemon
echo -e "${MAGENTA}Warming up Gradle daemon...${NC}"
for i in $(seq 1 $WARMUP_RUNS); do
    ./gradlew help --offline > /dev/null 2>&1
done
echo -e "${GREEN}✓ Daemon warmed up${NC}"

# Stop daemon before starting tests
./gradlew --stop > /dev/null 2>&1
sleep 2
echo ""

# ============================================
# Scenario 1: WITHOUT configuration cache
# ============================================
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Scenario 1: WITHOUT configuration cache${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Start fresh daemon for this scenario
echo -e "${MAGENTA}Starting fresh daemon for scenario 1...${NC}"
./gradlew help --offline --no-configuration-cache > /dev/null 2>&1
echo -e "${GREEN}✓ Daemon ready${NC}\n"

# Clean builds
echo -e "${CYAN}Testing clean builds...${NC}"
for i in $(seq 1 $RUNS); do
    echo -ne "  Run $i/$RUNS: "
    start=$(date +%s.%N)
    ./gradlew clean build -x test --offline --no-configuration-cache > /tmp/gradle_no_cache_clean_$i.log 2>&1
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    durations_no_cache_clean+=($duration)
    echo -e "${BLUE}${duration}s${NC}"
done

# Incremental builds (no changes)
echo -e "\n${CYAN}Testing incremental builds (no changes)...${NC}"
./gradlew clean build -x test --offline --no-configuration-cache > /dev/null 2>&1
for i in $(seq 1 $RUNS); do
    echo -ne "  Run $i/$RUNS: "
    start=$(date +%s.%N)
    ./gradlew build -x test --offline --no-configuration-cache > /tmp/gradle_no_cache_$i.log 2>&1
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    durations_no_cache+=($duration)
    echo -e "${BLUE}${duration}s${NC}"
done

# Stop daemon after scenario 1
echo -e "\n${MAGENTA}Stopping daemon...${NC}"
./gradlew --stop > /dev/null 2>&1
sleep 2
echo -e "${GREEN}✓ Daemon stopped${NC}"

# ============================================
# Scenario 2: WITH configuration cache
# ============================================
echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Scenario 2: WITH configuration cache${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Clean configuration cache before starting
rm -rf .gradle

# Start fresh daemon for this scenario
echo -e "${MAGENTA}Starting fresh daemon for scenario 2...${NC}"
./gradlew help --offline > /dev/null 2>&1
echo -e "${GREEN}✓ Daemon ready${NC}\n"

# Clean builds (cache will miss - this is expected)
echo -e "${CYAN}Testing clean builds (cache miss expected)...${NC}"
for i in $(seq 1 $RUNS); do
    echo -ne "  Run $i/$RUNS: "
    # Run clean separately to avoid invalidating the cache
    ./gradlew clean --offline > /dev/null 2>&1
    start=$(date +%s.%N)
    ./gradlew build -x test --offline > /tmp/gradle_with_cache_clean_$i.log 2>&1
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    durations_with_cache_clean+=($duration)
    
    # Check cache status
    cache_status=$(check_cache_status "/tmp/gradle_with_cache_clean_$i.log")
    if [ "$cache_status" = "HIT" ]; then
        echo -e "${BLUE}${duration}s${NC} ${GREEN}[CACHE HIT]${NC}"
    elif [ "$cache_status" = "MISS" ]; then
        echo -e "${BLUE}${duration}s${NC} ${YELLOW}[CACHE MISS - expected for clean]${NC}"
    else
        echo -e "${BLUE}${duration}s${NC} ${YELLOW}[CACHE UNKNOWN]${NC}"
    fi
done

# Incremental builds (no changes) - cache should HIT here
echo -e "\n${CYAN}Testing incremental builds (no changes - cache hit expected)...${NC}"
# Do one build to populate the cache
./gradlew clean --offline > /dev/null 2>&1
./gradlew build -x test --offline > /dev/null 2>&1
for i in $(seq 1 $RUNS); do
    echo -ne "  Run $i/$RUNS: "
    start=$(date +%s.%N)
    ./gradlew build -x test --offline > /tmp/gradle_with_cache_$i.log 2>&1
    end=$(date +%s.%N)
    duration=$(echo "$end - $start" | bc)
    durations_with_cache+=($duration)
    
    # Check cache status
    cache_status=$(check_cache_status "/tmp/gradle_with_cache_$i.log")
    if [ "$cache_status" = "HIT" ]; then
        echo -e "${BLUE}${duration}s${NC} ${GREEN}[CACHE HIT ✓]${NC}"
        cache_hits+=($i)
    elif [ "$cache_status" = "MISS" ]; then
        echo -e "${BLUE}${duration}s${NC} ${RED}[CACHE MISS ✗ - unexpected!]${NC}"
        cache_misses+=($i)
    else
        echo -e "${BLUE}${duration}s${NC} ${YELLOW}[CACHE UNKNOWN]${NC}"
    fi
done

# Stop daemon after scenario 2
echo -e "\n${MAGENTA}Stopping daemon...${NC}"
./gradlew --stop > /dev/null 2>&1
sleep 2
echo -e "${GREEN}✓ Daemon stopped${NC}"

# ============================================
# Analyze configuration cache problems
# ============================================
echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${YELLOW}Configuration Cache Diagnostics${NC}"
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"

# Find the most recent configuration cache report
CACHE_REPORT=$(find build/reports/configuration-cache -name "configuration-cache-report.html" -type f 2>/dev/null | head -1)

if [ -n "$CACHE_REPORT" ]; then
    echo -e "${CYAN}Configuration Cache Report Location:${NC}"
    echo -e "  $CACHE_REPORT\n"
    
    # Try to extract problem count
    if command -v grep &> /dev/null; then
        PROBLEM_COUNT=$(grep -o '"diagnostics":\[' "$CACHE_REPORT" 2>/dev/null | wc -l | tr -d ' ')
        if [ "$PROBLEM_COUNT" -gt 0 ]; then
            echo -e "${RED}⚠️  Found configuration cache problems!${NC}"
            echo -e "   Open the report to see details:"
            echo -e "   ${CYAN}open '$CACHE_REPORT'${NC}\n"
        fi
    fi
else
    echo -e "${YELLOW}No configuration cache report found${NC}\n"
fi

# Check for common issues in logs
echo -e "${CYAN}Checking for common issues:${NC}"

# Check for listener registration issues
if grep -q "addProjectEvaluationListener" /tmp/gradle_with_cache_*.log 2>/dev/null; then
    echo -e "  ${RED}✗ Found project evaluation listener issue (likely Artifactory plugin)${NC}"
else
    echo -e "  ${GREEN}✓ No listener registration issues${NC}"
fi

# Check for file system access issues
if grep -q "file system entry" /tmp/gradle_with_cache_*.log 2>/dev/null; then
    echo -e "  ${RED}✗ Found file system access at configuration time${NC}"
else
    echo -e "  ${GREEN}✓ No file system access issues${NC}"
fi

# Check for task graph calculation
TASK_GRAPH_COUNT=$(grep -c "Calculating task graph" /tmp/gradle_with_cache_*.log 2>/dev/null || echo "0")
if [ "$TASK_GRAPH_COUNT" -gt 0 ]; then
    echo -e "  ${RED}✗ Task graph recalculated $TASK_GRAPH_COUNT times (cache not being reused)${NC}"
else
    echo -e "  ${GREEN}✓ Configuration cache being reused${NC}"
fi

# Show sample of actual cache-related log messages
echo -e "\n${CYAN}Sample cache messages from logs:${NC}"
grep -h -i "configuration cache\|task graph\|reusing\|calculating" /tmp/gradle_with_cache_*.log 2>/dev/null | head -10 | sed 's/^/  /'

echo ""

# ============================================
# Calculate statistics
# ============================================

calculate_stats() {
    local array_name=$1
    eval "local arr=(\"\${${array_name}[@]}\")"

    local sum=0
    local min=${arr[0]}
    local max=${arr[0]}

    for val in "${arr[@]}"; do
        sum=$(echo "$sum + $val" | bc)
        if (( $(echo "$val < $min" | bc -l) )); then
            min=$val
        fi
        if (( $(echo "$val > $max" | bc -l) )); then
            max=$val
        fi
    done

    local count=${#arr[@]}
    local avg=$(echo "scale=3; $sum / $count" | bc)

    # Calculate median
    local sorted_str=$(printf '%s\n' "${arr[@]}" | sort -n | tr '\n' ' ')
    local sorted=($sorted_str)
    local median
    if (( count % 2 == 0 )); then
        local mid1=${sorted[$((count/2-1))]}
        local mid2=${sorted[$((count/2))]}
        median=$(echo "scale=3; ($mid1 + $mid2) / 2" | bc)
    else
        median=${sorted[$((count/2))]}
    fi

    # Calculate standard deviation
    local variance=0
    for val in "${arr[@]}"; do
        local diff=$(echo "$val - $avg" | bc)
        local sq=$(echo "$diff * $diff" | bc)
        variance=$(echo "$variance + $sq" | bc)
    done
    variance=$(echo "scale=6; $variance / $count" | bc)
    local stddev=$(echo "scale=3; sqrt($variance)" | bc)

    echo "$avg $median $min $max $stddev"
}

stats_no_cache_clean=$(calculate_stats "durations_no_cache_clean")
stats_with_cache_clean=$(calculate_stats "durations_with_cache_clean")
stats_no_cache=$(calculate_stats "durations_no_cache")
stats_with_cache=$(calculate_stats "durations_with_cache")

read avg_no_cache_clean med_no_cache_clean min_no_cache_clean max_no_cache_clean std_no_cache_clean <<< "$stats_no_cache_clean"
read avg_with_cache_clean med_with_cache_clean min_with_cache_clean max_with_cache_clean std_with_cache_clean <<< "$stats_with_cache_clean"
read avg_no_cache med_no_cache min_no_cache max_no_cache std_no_cache <<< "$stats_no_cache"
read avg_with_cache med_with_cache min_with_cache max_with_cache std_with_cache <<< "$stats_with_cache"

# ============================================
# Print comprehensive results
# ============================================

echo -e "\n${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                   BENCHMARK RESULTS                    ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}\n"

print_comparison() {
    local title=$1
    local avg1=$2
    local med1=$3
    local min1=$4
    local max1=$5
    local std1=$6
    local avg2=$7
    local med2=$8
    local min2=$9
    local max2=${10}
    local std2=${11}

    echo -e "${YELLOW}${title}${NC}"
    echo -e "${CYAN}┌─────────────────────────┬──────────────┬──────────────┐${NC}"
    echo -e "${CYAN}│ Metric                  │ Without Cache│ With Cache   │${NC}"
    echo -e "${CYAN}├─────────────────────────┼──────────────┼──────────────┤${NC}"
    printf "${CYAN}│${NC} %-23s ${CYAN}│${NC} %11.3fs ${CYAN}│${NC} %11.3fs ${CYAN}│${NC}\n" "Average" "$avg1" "$avg2"
    printf "${CYAN}│${NC} %-23s ${CYAN}│${NC} %11.3fs ${CYAN}│${NC} %11.3fs ${CYAN}│${NC}\n" "Median" "$med1" "$med2"
    printf "${CYAN}│${NC} %-23s ${CYAN}│${NC} %11.3fs ${CYAN}│${NC} %11.3fs ${CYAN}│${NC}\n" "Min" "$min1" "$min2"
    printf "${CYAN}│${NC} %-23s ${CYAN}│${NC} %11.3fs ${CYAN}│${NC} %11.3fs ${CYAN}│${NC}\n" "Max" "$max1" "$max2"
    printf "${CYAN}│${NC} %-23s ${CYAN}│${NC} %11.3fs ${CYAN}│${NC} %11.3fs ${CYAN}│${NC}\n" "Std Deviation" "$std1" "$std2"
    echo -e "${CYAN}└─────────────────────────┴──────────────┴──────────────┘${NC}"

    local diff=$(echo "scale=3; $avg1 - $avg2" | bc)
    local abs_diff=$(echo "$diff" | tr -d '-')

    if (( $(echo "$diff > 0" | bc -l) )); then
        local percent=$(echo "scale=1; ($diff / $avg1) * 100" | bc)
        echo -e "${GREEN}⚡ Configuration cache is ${abs_diff}s faster (${percent}% improvement)${NC}\n"
    elif (( $(echo "$diff < 0" | bc -l) )); then
        local percent=$(echo "scale=1; ($abs_diff / $avg2) * 100" | bc)
        echo -e "${RED}⚠ Configuration cache is ${abs_diff}s slower (${percent}% regression)${NC}\n"
    else
        echo -e "${YELLOW}➡ No significant difference${NC}\n"
    fi
}

print_comparison "📦 Clean Build Performance" \
    "$avg_no_cache_clean" "$med_no_cache_clean" "$min_no_cache_clean" "$max_no_cache_clean" "$std_no_cache_clean" \
    "$avg_with_cache_clean" "$med_with_cache_clean" "$min_with_cache_clean" "$max_with_cache_clean" "$std_with_cache_clean"

print_comparison "🔄 Incremental Build Performance (No Changes)" \
    "$avg_no_cache" "$med_no_cache" "$min_no_cache" "$max_no_cache" "$std_no_cache" \
    "$avg_with_cache" "$med_with_cache" "$min_with_cache" "$max_with_cache" "$std_with_cache"

# Cache hit/miss statistics
if [ ${#cache_hits[@]} -gt 0 ] || [ ${#cache_misses[@]} -gt 0 ]; then
    echo -e "${YELLOW}📊 Cache Hit/Miss Statistics${NC}"
    echo -e "${CYAN}┌─────────────────────────┬──────────────┐${NC}"
    echo -e "${CYAN}│ Metric                  │ Count        │${NC}"
    echo -e "${CYAN}├─────────────────────────┼──────────────┤${NC}"
    printf "${CYAN}│${NC} %-23s ${CYAN}│${NC} %12d ${CYAN}│${NC}\n" "Cache Hits" "${#cache_hits[@]}"
    printf "${CYAN}│${NC} %-23s ${CYAN}│${NC} %12d ${CYAN}│${NC}\n" "Cache Misses" "${#cache_misses[@]}"
    
    total_cache_tests=$((${#cache_hits[@]} + ${#cache_misses[@]}))
    if [ $total_cache_tests -gt 0 ]; then
        hit_rate=$(echo "scale=1; ${#cache_hits[@]} * 100 / $total_cache_tests" | bc)
        printf "${CYAN}│${NC} %-23s ${CYAN}│${NC} %11.1f%% ${CYAN}│${NC}\n" "Hit Rate" "$hit_rate"
    fi
    echo -e "${CYAN}└─────────────────────────┴──────────────┘${NC}\n"
fi

# ============================================
# Summary and recommendations
# ============================================

echo -e "${BLUE}╔════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                        SUMMARY                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════╝${NC}\n"

total_diff_clean=$(echo "scale=3; $avg_no_cache_clean - $avg_with_cache_clean" | bc)
total_diff_incr=$(echo "scale=3; $avg_no_cache - $avg_with_cache" | bc)
total_diff=$(echo "scale=3; $total_diff_clean + $total_diff_incr" | bc)

if (( $(echo "$total_diff > 0" | bc -l) )); then
    echo -e "${GREEN}✅ Configuration cache shows overall performance improvement${NC}"
    echo -e "   Total time saved per build cycle: ${GREEN}${total_diff}s${NC}\n"

    # Calculate potential daily/weekly savings
    daily_builds=10
    weekly_savings=$(echo "scale=1; $total_diff * $daily_builds * 5 / 60" | bc)
    echo -e "${CYAN}💡 Potential time savings:${NC}"
    echo -e "   • Per build cycle: ${total_diff}s"
    echo -e "   • Per day (~${daily_builds} builds): $(echo "scale=1; $total_diff * $daily_builds" | bc)s"
    echo -e "   • Per week: ${weekly_savings} minutes\n"

    echo -e "${GREEN}📊 Recommendation: ENABLE configuration cache${NC}"
else
    echo -e "${RED}⚠️  Configuration cache shows performance regression${NC}"
    echo -e "   Total time lost per build cycle: ${RED}$(echo "$total_diff" | tr -d '-')s${NC}\n"
    
    echo -e "${YELLOW}🔍 Likely causes:${NC}"
    echo -e "   • Cache is being invalidated on every build (not reused)"
    echo -e "   • Incompatible plugins (check Artifactory plugin)"
    echo -e "   • Configuration-time file access (local.properties)"
    echo -e "   • Dynamic script loading from remote URLs\n"
    
    echo -e "${CYAN}📋 Next steps:${NC}"
    echo -e "   1. Review CONFIG_CACHE_ANALYSIS.md for detailed analysis"
    echo -e "   2. Open configuration cache report: ${CYAN}open '$CACHE_REPORT'${NC}"
    echo -e "   3. Fix Artifactory plugin compatibility"
    echo -e "   4. Update gradle.properties to use provider API for file access"
    echo -e "   5. Re-run this benchmark\n"
    
    echo -e "${RED}📊 Recommendation: Keep configuration cache DISABLED until issues are fixed${NC}"
fi

echo -e "\n${BLUE}╚════════════════════════════════════════════════════════╝${NC}"

# Cleanup temp files
rm -f /tmp/gradle_*.log



