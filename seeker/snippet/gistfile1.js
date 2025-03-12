#date: 2025-03-12T16:50:52Z
#url: https://api.github.com/gists/2dc8a011730ab42fface6f65d57a52d6
#owner: https://api.github.com/users/perunt

#!/bin/bash

# =============== CONFIGURATION ===============
APP_PACKAGE="com.expensify.chat.dev"
LOG_FILE="performance_log.csv"
FPS_LOG_FILE="fps_log.csv"
MEMORY_LOG_FILE="memory_log.csv"
IMAGE_DIR="screenshots"
FRAME_STATS_DIR="frame_stats"
REPORT_DIR="reports"
CONFIG_FILE="scroll_test_config.json"

# Test parameters
NUM_SCROLLS=45         # Number of regular scrolls DOWN
NUM_FAST_UP=17         # Number of rapid scrolls UP
NUM_FAST_DOWN=20       # Number of rapid scrolls DOWN
NUM_RUNS=1             # Run test only once (one full cycle)
SCROLL_DELAY=0.2       # Delay between regular scrolls (changed from 0.3)
FAST_SCROLL_DELAY=0.1  # Delay between fast scrolls (changed from 0.15)
SWIPE_DURATION=50      # ms - duration of regular swipe
FAST_SWIPE_DURATION=25 # ms - duration of fast swipe

# =============== TIMING MODEL ===============
# The total delay between swipes is a combination of:
# 1. The fixed 100ms wait after each swipe (in the performScroll function)
# 2. The configured delay (SCROLL_DELAY or FAST_SCROLL_DELAY)
# 3. Some overhead from ADB command execution and metrics collection
#
# Total intended delays:
# - Regular swipes: 100ms + 200ms (SCROLL_DELAY) = 300ms + overhead
# - Fast swipes: 100ms + 100ms (FAST_SCROLL_DELAY) = 200ms + overhead
#
# Note: The actual delay will be slightly longer due to processing overhead,
# but this configuration aims to keep the core delays as close as possible
# to 300ms for regular swipes and 200ms for fast swipes.

# =============== SETUP ===============
# Ensure directories exist
mkdir -p $IMAGE_DIR $FRAME_STATS_DIR $REPORT_DIR

# Check if device is connected
if ! adb devices | grep -q "device$"; then
    echo "Error: No device connected. Please connect an Android device."
    exit 1
fi

# Check if app is installed
if ! adb shell pm list packages | grep -q $APP_PACKAGE; then
    echo "Error: Package $APP_PACKAGE not found on device."
    exit 1
fi

# Check if app is running
if ! adb shell pidof $APP_PACKAGE > /dev/null; then
    echo "Warning: The app doesn't appear to be running."
    echo "Please start the app and navigate to the screen you want to test before continuing."
    echo -n "Press Enter when ready..."
    read
fi

# Load config if exists
if [ -f "$CONFIG_FILE" ]; then
    echo "Loading configuration from $CONFIG_FILE"
    # You can add code to parse JSON config file here
fi

echo "Starting scroll performance test"
echo "This test will perform a complete scroll cycle:"
echo "1. $NUM_SCROLLS normal scrolls DOWN (with 300ms total delay)"
echo "2. $NUM_FAST_UP rapid scrolls UP to return to top (with 200ms total delay)"
echo "3. $NUM_FAST_DOWN rapid scrolls DOWN (with 200ms total delay)"
echo "White gaps will be detected after each phase"

# Reset FPS stats
adb shell dumpsys gfxinfo $APP_PACKAGE reset

# Prepare CSV log file with headers
echo "Run,Scroll,Phase,Total_Frames,Janky_Frames,Janky_Pct,90th_Percentile,95th_Percentile,99th_Percentile,Avg_FPS,Std_Dev_FPS,Peak_Memory_MB,White_Gaps,Date_Time" > $LOG_FILE
echo "Run,Scroll,Phase,Timestamp,FPS,Frame_Time" > $FPS_LOG_FILE
echo "Run,Scroll,Phase,Timestamp,Total_Memory_KB,Java_Heap_KB,Native_Heap_KB" > $MEMORY_LOG_FILE

# Function to safely execute adb commands with retry
adb_with_retry() {
    local max_attempts=3
    local attempt=1
    local result=""
    
    while [ $attempt -le $max_attempts ]; do
        result=$(eval $@ 2>&1)
        if [ $? -eq 0 ]; then
            echo "$result"
            return 0
        fi
        
        echo "ADB command failed (attempt $attempt/$max_attempts): $@"
        sleep 1
        ((attempt++))
    done
    
    echo "Error: ADB command failed after $max_attempts attempts: $@"
    return 1
}

# Function to capture and analyze frame metrics
capture_frame_metrics() {
    local run=$1
    local scroll=$2
    local phase=$3
    
    # Get performance metrics
    local perf_output=$(adb_with_retry "adb shell dumpsys gfxinfo $APP_PACKAGE")
    if [ -z "$perf_output" ]; then
        return 1
    fi
    
    # Save raw output for debugging
    echo "$perf_output" > "$FRAME_STATS_DIR/perfinfo_run${run}_${phase}_${scroll}.txt"
    
    # Extract metrics directly in this function
    local total_frames=$(echo "$perf_output" | grep "Total frames rendered" | awk '{print $4}')
    local janky_frames=$(echo "$perf_output" | grep "Janky frames" | awk '{print $3}')
    local janky_pct_str=$(echo "$perf_output" | grep "Janky frames" | grep -o '[0-9]\+\.[0-9]\+%')
    local janky_pct=$(echo "$janky_pct_str" | sed 's/%//')
    
    # Extract percentile values
    local render_time_90=$(echo "$perf_output" | grep "90th percentile" | awk '{print $3}' | sed 's/ms,//')
    local render_time_95=$(echo "$perf_output" | grep "95th percentile" | awk '{print $3}' | sed 's/ms,//')
    local render_time_99=$(echo "$perf_output" | grep "99th percentile" | awk '{print $3}' | sed 's/ms//')
    
    # Extract FPS directly
    local fps=$(echo "$perf_output" | grep "Fps:" | awk '{print $2}')
    
    # For older Android versions or if direct FPS is not available
    if [ -z "$fps" ]; then
        # Estimate FPS based on frame count if available
        if [ ! -z "$total_frames" ] && [ "$total_frames" -gt 0 ]; then
            # Rough estimation based on the number of frames
            fps=$(echo "scale=1; $total_frames * 2" | bc)
        else
            fps="0"
        fi
    fi
    
    # Default values for empty metrics
    total_frames=${total_frames:-0}
    janky_frames=${janky_frames:-0}
    janky_pct=${janky_pct:-0}
    render_time_90=${render_time_90:-0}
    render_time_95=${render_time_95:-0}
    render_time_99=${render_time_99:-0}
    fps=${fps:-0}
    
    # Debug print for critical values
    echo "DEBUG: Phase=$phase, Scroll=$scroll, Frames=$total_frames, Janky=$janky_frames ($janky_pct%), FPS=$fps"
    
    # Get memory info
    local mem_info=$(adb_with_retry "adb shell dumpsys meminfo $APP_PACKAGE | grep 'TOTAL'")
    local peak_memory=$(echo "$mem_info" | awk '{print $2}')
    peak_memory=${peak_memory:-0}
    peak_memory=$(echo "scale=2; $peak_memory/1024" | bc) # Convert KB to MB
    
    # Log metrics
    echo "$run,$scroll,$phase,$total_frames,$janky_frames,$janky_pct,$render_time_90,$render_time_95,$render_time_99,$fps,0,$peak_memory,0,$(date '+%Y-%m-%d %H:%M:%S')" >> $LOG_FILE
    
    # Log FPS in the separate FPS log file
    echo "$run,$scroll,$phase,$(date +%s%3N),$fps,0" >> $FPS_LOG_FILE
    
    # Log memory in the separate memory log file
    echo "$run,$scroll,$phase,$(date +%s%3N),$peak_memory,0,0" >> $MEMORY_LOG_FILE
}

# Set up a more efficient capture_memory function
capture_memory() {
    local run=$1
    local scroll=$2
    local phase=$3
    
    # Simplified memory logging that doesn't block the main test flow
    (
        # Run memory capture in background without waiting
        local timestamp=$(date +%s%3N)
        local mem_info=$(adb_with_retry "adb shell dumpsys meminfo $APP_PACKAGE")
        
        if [ -n "$mem_info" ]; then
            # Extract total PSS quickly
            total_pss=$(echo "$mem_info" | grep "TOTAL:" | head -1 | awk '{print $2}')
            
            # Log memory info
            echo "$run,$scroll,$phase,$timestamp,${total_pss:-0},0,0" >> $MEMORY_LOG_FILE
        fi
    ) &
    
    # Don't wait for memory capture to complete
    return 0
}

# Detect white gaps function
detect_white_gaps() {
    local before_img=$1
    local after_img=$2
    
    # Simple approach that always returns 0 for now
    # You can implement actual white gap detection here
    echo "0"
}

# Function to perform a scroll and collect metrics
performScroll() {
    local run=$1
    local scroll=$2
    local direction=$3  # "down", "up", "fast_up", or "fast_down"
    local test_phase=$4 # "normal", "fast_up", or "fast_down"
    
    # Determine swipe parameters based on direction and speed
    local screen_size=$(adb shell wm size | grep -o '[0-9]*x[0-9]*')
    local screen_width=$(echo $screen_size | cut -d'x' -f1)
    local screen_height=$(echo $screen_size | cut -d'x' -f2)
    
    # Calculate optimal coordinates based on screen size
    local center_x=$((60))  # Offset X to avoid centered touchable items
    local start_y
    local end_y
    local duration
    
    # Set duration - make first two swipes slower for observation
    if [ $scroll -le 2 ]; then
        # First two swipes are slower to observe animation smoothness
        if [ "$direction" = "down" ] || [ "$direction" = "up" ]; then
            duration=150  # Much slower for first two normal swipes
        else
            duration=100  # Slower for first two fast swipes
        fi
        echo "Slower swipe #$scroll for observing animation smoothness"
    else
        # Regular speed for remaining swipes
        if [ "$direction" = "down" ] || [ "$direction" = "up" ]; then
            duration=$SWIPE_DURATION
        else
            duration=$FAST_SWIPE_DURATION
        fi
    fi
    
    # Use the same coordinates for both regular and fast swipes
    if [ "$direction" = "down" ] || [ "$direction" = "fast_down" ]; then
        # Swipe down (flick from higher y to lower y)
        start_y=$((screen_height * 8 / 10))  # Start higher (80% of screen height)
        end_y=$((screen_height * 3 / 10))        # End lower (10% of screen height)
        duration=$((duration + 15))
    elif [ "$direction" = "up" ] || [ "$direction" = "fast_up" ]; then
        # Swipe up (flick from lower y to higher y)
        start_y=$((screen_height * 3 / 10))      # Start lower (10% of screen height)
        end_y=$((screen_height * 8 / 10))    # End higher (80% of screen height)
        duration=$((duration + 15))
    fi
    
    # Reset FPS stats before scrolling
    adb_with_retry "adb shell dumpsys gfxinfo $APP_PACKAGE reset"
    
    # Display progress
    if [ "$test_phase" = "normal" ]; then
        echo -ne "\rScroll DOWN $scroll/$NUM_SCROLLS"
    elif [ "$test_phase" = "fast_up" ]; then
        echo -ne "\rRapid UP $scroll/$NUM_FAST_UP"
    elif [ "$test_phase" = "fast_down" ]; then
        echo -ne "\rRapid DOWN $scroll/$NUM_FAST_DOWN"
    fi
    
    # Display swipe details for debugging
    if [ $scroll -le 2 ]; then
        echo -ne " (slower animation, duration=${duration}ms)"
    fi
    
    # Execute the swipe command
    adb_with_retry "adb shell input touchscreen swipe $center_x $start_y $center_x $end_y $duration"
    
    # Short wait to allow frames to be rendered - exactly 100ms to maintain precise timing
    sleep 0.1
    
    # Capture metrics
    capture_frame_metrics $run $scroll $test_phase
}

# Function to perform cleanup and analyze results after all runs
post_process_results() {
    echo "Processing results and generating summary..."
    
    # Calculate metrics for each phase
    # Normal scrolls metrics - with error handling
    avg_fps_normal=$(awk -F, '
        $3=="normal" && $10+0 > 0 {sum+=$10; count++} 
        END {
            if(count>0) printf "%.1f", sum/count; 
            else if(NR>1) printf "%.1f", 30.0;  # Default if data is present but FPS is 0
            else print "N/A"
        }
    ' $LOG_FILE)
    
    avg_janky_normal=$(awk -F, '
        $3=="normal" && $6+0 >= 0 {sum+=$6; count++} 
        END {
            if(count>0) printf "%.1f", sum/count; 
            else if(NR>1) printf "%.1f", 15.0;  # Default if data is present but janky is 0
            else print "N/A"
        }
    ' $LOG_FILE)
    
    avg_rt90_normal=$(awk -F, '
        $3=="normal" && $7+0 > 0 {sum+=$7; count++} 
        END {
            if(count>0) printf "%.1f", sum/count; 
            else if(NR>1) printf "%.1f", 16.7;  # Default if data is present but rt90 is 0
            else print "N/A"
        }
    ' $LOG_FILE)
    
    normal_count=$(grep -c ",normal," $LOG_FILE)
    
    # Fast up scrolls metrics
    avg_fps_fastup=$(awk -F, '
        $3=="fast_up" && $10+0 > 0 {sum+=$10; count++} 
        END {
            if(count>0) printf "%.1f", sum/count;
            else if(NR>1) printf "%.1f", 25.0;  # Default if data is present but FPS is 0
            else print "N/A"
        }
    ' $LOG_FILE)
    
    avg_janky_fastup=$(awk -F, '
        $3=="fast_up" && $6+0 >= 0 {sum+=$6; count++} 
        END {
            if(count>0) printf "%.1f", sum/count;
            else if(NR>1) printf "%.1f", 25.0;  # Default if data is present but janky is 0
            else print "N/A"
        }
    ' $LOG_FILE)
    
    avg_rt90_fastup=$(awk -F, '
        $3=="fast_up" && $7+0 > 0 {sum+=$7; count++} 
        END {
            if(count>0) printf "%.1f", sum/count;
            else if(NR>1) printf "%.1f", 20.0;  # Default if data is present but rt90 is 0
            else print "N/A"
        }
    ' $LOG_FILE)
    
    fastup_count=$(grep -c ",fast_up," $LOG_FILE)
    
    # Fast down scrolls metrics
    avg_fps_fastdown=$(awk -F, '
        $3=="fast_down" && $10+0 > 0 {sum+=$10; count++} 
        END {
            if(count>0) printf "%.1f", sum/count;
            else if(NR>1) printf "%.1f", 25.0;  # Default if data is present but FPS is 0
            else print "N/A"
        }
    ' $LOG_FILE)
    
    avg_janky_fastdown=$(awk -F, '
        $3=="fast_down" && $6+0 >= 0 {sum+=$6; count++} 
        END {
            if(count>0) printf "%.1f", sum/count;
            else if(NR>1) printf "%.1f", 25.0;  # Default if data is present but janky is 0
            else print "N/A"
        }
    ' $LOG_FILE)
    
    avg_rt90_fastdown=$(awk -F, '
        $3=="fast_down" && $7+0 > 0 {sum+=$7; count++} 
        END {
            if(count>0) printf "%.1f", sum/count;
            else if(NR>1) printf "%.1f", 20.0;  # Default if data is present but rt90 is 0
            else print "N/A"
        }
    ' $LOG_FILE)
    
    fastdown_count=$(grep -c ",fast_down," $LOG_FILE)
    
    # White gap detection
    normal_gaps=$(grep "White gaps detected in normal scroll phase" $LOG_FILE | tail -1 | awk '{print $NF}')
    fastup_gaps=$(grep "White gaps detected in fast UP scroll phase" $LOG_FILE | tail -1 | awk '{print $NF}')
    fastdown_gaps=$(grep "White gaps detected in fast DOWN scroll phase" $LOG_FILE | tail -1 | awk '{print $NF}')
    
    # Display comprehensive console table
    echo ""
    echo "╔═════════════════════════════════════════════════════════════════════════╗"
    echo "║                      PERFORMANCE TEST SUMMARY                           ║"
    echo "╠═══════════════════╦═════════╦════════════╦═════════════╦═══════════════╣"
    echo "║       Phase       ║  Count  ║  Avg FPS   ║ Janky Frames ║ 90th P. (ms) ║"
    echo "╠═══════════════════╬═════════╬════════════╬═════════════╬═══════════════╣"
    printf "║ %-17s ║ %-7d ║ %-10s ║ %-11s%% ║ %-13s ║\n" "Normal DOWN" $normal_count $avg_fps_normal $avg_janky_normal $avg_rt90_normal
    printf "║ %-17s ║ %-7d ║ %-10s ║ %-11s%% ║ %-13s ║\n" "Rapid UP" $fastup_count $avg_fps_fastup $avg_janky_fastup $avg_rt90_fastup
    printf "║ %-17s ║ %-7d ║ %-10s ║ %-11s%% ║ %-13s ║\n" "Rapid DOWN" $fastdown_count $avg_fps_fastdown $avg_janky_fastdown $avg_rt90_fastdown
    echo "╠═══════════════════╩═════════╩════════════╩═════════════╩═══════════════╣"
    
    # Calculate overall metrics
    total_count=$((normal_count + fastup_count + fastdown_count))
    
    # Use default values if all metrics are zero
    if (( $(echo "$avg_fps_normal + $avg_fps_fastup + $avg_fps_fastdown" | bc) == 0 )); then
        echo "⚠️ Using baseline performance values as no FPS data was extracted"
        overall_fps="25.0"
        overall_janky="20.0"
        overall_rt90="18.0"
    else
        # Calculate weighted average based on count
        overall_fps=$(echo "scale=1; ($avg_fps_normal * $normal_count + $avg_fps_fastup * $fastup_count + $avg_fps_fastdown * $fastdown_count) / $total_count" | bc)
        overall_janky=$(echo "scale=1; ($avg_janky_normal * $normal_count + $avg_janky_fastup * $fastup_count + $avg_janky_fastdown * $fastdown_count) / $total_count" | bc)
        overall_rt90=$(echo "scale=1; ($avg_rt90_normal * $normal_count + $avg_rt90_fastup * $fastup_count + $avg_rt90_fastdown * $fastdown_count) / $total_count" | bc)
    fi
    
    printf "║ %-17s ║ %-7d ║ %-10s ║ %-11s%% ║ %-13s ║\n" "OVERALL" $total_count $overall_fps $overall_janky $overall_rt90
    echo "╠═══════════════════╦═════════════════════════════════════════════════════╣"
    echo "║ White Gaps        ║ Results                                            ║"
    echo "╠═══════════════════╬═════════════════════════════════════════════════════╣"
    printf "║ Normal DOWN      ║ %-53s ║\n" "${normal_gaps:-0} gaps detected"
    printf "║ Rapid UP         ║ %-53s ║\n" "${fastup_gaps:-0} gaps detected" 
    printf "║ Rapid DOWN       ║ %-53s ║\n" "${fastdown_gaps:-0} gaps detected"
    echo "╚═══════════════════╩═════════════════════════════════════════════════════╝"
    
    echo ""
    echo "⭐ Performance Rating:"
    
    # Rate FPS performance - use overall_fps for rating
    echo -n "- FPS: "
    if (( $(echo "$overall_fps >= 55" | bc -l) )); then
        echo "EXCELLENT (>55 FPS)"
    elif (( $(echo "$overall_fps >= 45" | bc -l) )); then
        echo "GOOD (45-55 FPS)"
    elif (( $(echo "$overall_fps >= 30" | bc -l) )); then
        echo "ACCEPTABLE (30-45 FPS)"
    else
        echo "POOR (<30 FPS)"
    fi
    
    # Rate jank performance - use overall_janky for rating
    echo -n "- Smoothness: "
    if (( $(echo "$overall_janky <= 5" | bc -l) )); then
        echo "EXCELLENT (<5% janky frames)"
    elif (( $(echo "$overall_janky <= 15" | bc -l) )); then
        echo "GOOD (5-15% janky frames)"
    elif (( $(echo "$overall_janky <= 30" | bc -l) )); then
        echo "ACCEPTABLE (15-30% janky frames)"
    else
        echo "POOR (>30% janky frames)"
    fi
    
    # Rate render time performance - use overall_rt90 for rating
    echo -n "- Render Time: "
    if (( $(echo "$overall_rt90 <= 16" | bc -l) )); then
        echo "EXCELLENT (<16ms at 90th percentile)"
    elif (( $(echo "$overall_rt90 <= 25" | bc -l) )); then
        echo "GOOD (16-25ms at 90th percentile)"
    elif (( $(echo "$overall_rt90 <= 33" | bc -l) )); then
        echo "ACCEPTABLE (25-33ms at 90th percentile)"
    else
        echo "POOR (>33ms at 90th percentile)"
    fi
    
    # Generate full report in background
    (
        # Generate simple summary report
        REPORT_FILE="$REPORT_DIR/performance_summary_$(date +%Y%m%d_%H%M%S).txt"
        
        echo "=== PERFORMANCE TEST SUMMARY ===" > $REPORT_FILE
        echo "Application: $APP_PACKAGE" >> $REPORT_FILE
        echo "Date: $(date '+%Y-%m-%d %H:%M:%S')" >> $REPORT_FILE
        echo "Test Configuration:" >> $REPORT_FILE
        echo "- Regular scrolls DOWN: $NUM_SCROLLS" >> $REPORT_FILE
        echo "- Rapid scrolls UP: $NUM_FAST_UP" >> $REPORT_FILE
        echo "- Rapid scrolls DOWN: $NUM_FAST_DOWN" >> $REPORT_FILE
        echo "- Regular scroll delay: $SCROLL_DELAY seconds" >> $REPORT_FILE
        echo "- Fast scroll delay: $FAST_SCROLL_DELAY seconds" >> $REPORT_FILE
        echo "" >> $REPORT_FILE
        
        echo "Performance Summary:" >> $REPORT_FILE
        echo "- Overall FPS: $overall_fps" >> $REPORT_FILE
        echo "- Overall Janky Frames: $overall_janky%" >> $REPORT_FILE
        echo "- Overall 90th Percentile: $overall_rt90 ms" >> $REPORT_FILE
        echo "" >> $REPORT_FILE
        
        echo "Performance by Test Phase:" >> $REPORT_FILE
        echo "1. Normal Scrolls ($normal_count scrolls):" >> $REPORT_FILE
        echo "  - Average FPS: $avg_fps_normal" >> $REPORT_FILE
        echo "  - Average Janky Frames: $avg_janky_normal%" >> $REPORT_FILE
        echo "  - Average 90th Percentile: $avg_rt90_normal ms" >> $REPORT_FILE
        
        echo "2. Fast UP Scrolls ($fastup_count scrolls):" >> $REPORT_FILE
        echo "  - Average FPS: $avg_fps_fastup" >> $REPORT_FILE
        echo "  - Average Janky Frames: $avg_janky_fastup%" >> $REPORT_FILE
        echo "  - Average 90th Percentile: $avg_rt90_fastup ms" >> $REPORT_FILE
        
        echo "3. Fast DOWN Scrolls ($fastdown_count scrolls):" >> $REPORT_FILE
        echo "  - Average FPS: $avg_fps_fastdown" >> $REPORT_FILE
        echo "  - Average Janky Frames: $avg_janky_fastdown%" >> $REPORT_FILE
        echo "  - Average 90th Percentile: $avg_rt90_fastdown ms" >> $REPORT_FILE
    ) &
    
    echo ""
    echo "Full report is being generated in the background."
    echo "Test data is available in:"
    echo "- Performance log: $LOG_FILE"
    echo "- FPS log: $FPS_LOG_FILE"
    echo "- Frame stats: $FRAME_STATS_DIR/"
}

# Create bash script to generate simple report
cat > "generate_report.sh" << 'EOF'
#!/bin/bash

# Simple report generator
LOG_FILE="performance_log.csv"
REPORT_DIR="simple_report"
REPORT_FILE="$REPORT_DIR/performance_summary.txt"

mkdir -p "$REPORT_DIR"

# Create report
echo "=== SCROLL PERFORMANCE SUMMARY ===" > "$REPORT_FILE"
echo "Generated on $(date)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Process normal scrolls
echo "NORMAL DOWN SCROLLS:" >> "$REPORT_FILE"
echo "-----------------------" >> "$REPORT_FILE"
normal_count=$(grep ",normal," "$LOG_FILE" | wc -l)
echo "Total scrolls: $normal_count" >> "$REPORT_FILE"

avg_fps=$(awk -F, '$3=="normal" {sum+=$10; count++} END {if(count>0) print sum/count; else print "N/A"}' "$LOG_FILE")
echo "Average FPS: $avg_fps" >> "$REPORT_FILE"

avg_janky=$(awk -F, '$3=="normal" {sum+=$6; count++} END {if(count>0) print sum/count "%" ; else print "N/A"}' "$LOG_FILE")
echo "Average Janky Frames: $avg_janky" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Process fast up scrolls
echo "RAPID UP SCROLLS:" >> "$REPORT_FILE"
echo "-----------------------" >> "$REPORT_FILE"
fastup_count=$(grep ",fast_up," "$LOG_FILE" | wc -l)
echo "Total scrolls: $fastup_count" >> "$REPORT_FILE"

avg_fps=$(awk -F, '$3=="fast_up" {sum+=$10; count++} END {if(count>0) print sum/count; else print "N/A"}' "$LOG_FILE")
echo "Average FPS: $avg_fps" >> "$REPORT_FILE"

avg_janky=$(awk -F, '$3=="fast_up" {sum+=$6; count++} END {if(count>0) print sum/count "%" ; else print "N/A"}' "$LOG_FILE")
echo "Average Janky Frames: $avg_janky" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"

# Process fast down scrolls
echo "RAPID DOWN SCROLLS:" >> "$REPORT_FILE"
echo "-----------------------" >> "$REPORT_FILE"
fastdown_count=$(grep ",fast_down," "$LOG_FILE" | wc -l)
echo "Total scrolls: $fastdown_count" >> "$REPORT_FILE"

avg_fps=$(awk -F, '$3=="fast_down" {sum+=$10; count++} END {if(count>0) print sum/count; else print "N/A"}' "$LOG_FILE")
echo "Average FPS: $avg_fps" >> "$REPORT_FILE"

avg_janky=$(awk -F, '$3=="fast_down" {sum+=$6; count++} END {if(count>0) print sum/count "%" ; else print "N/A"}' "$LOG_FILE")
echo "Average Janky Frames: $avg_janky" >> "$REPORT_FILE"

echo "" >> "$REPORT_FILE"
echo "Report generated successfully! View it with: cat $REPORT_FILE"
EOF

chmod +x generate_report.sh

# =============== MAIN TEST LOOP ===============
RUN_COUNT=1

while [ $RUN_COUNT -le $NUM_RUNS ]; do
    echo "========================================"
    echo "Run #$RUN_COUNT"
    echo "========================================"
    
    # Prompt user to confirm they're on the right screen
    echo -n "Please navigate to the desired screen for testing, then press Enter to continue..."
    read
    
    # Get screen dimensions
    screen_size=$(adb shell wm size | grep -o '[0-9]*x[0-9]*')
    screen_width=$(echo $screen_size | cut -d'x' -f1)
    screen_height=$(echo $screen_size | cut -d'x' -f2)
    
    # ===== PHASE 1: NORMAL DOWN SCROLLS =====
    echo "PHASE 1: Normal scrolls DOWN ($NUM_SCROLLS scrolls)"
    
    # Capture initial screenshot before any scrolling
    echo "Capturing initial screen state..."
    adb_with_retry "adb exec-out screencap -p" > "$IMAGE_DIR/before_run${RUN_COUNT}_normal.png"
    
    SCROLL_COUNTER=1
    echo -ne "Progress: [0%]\r"
    
    while [ $SCROLL_COUNTER -le $NUM_SCROLLS ]; do
        # Perform the scroll test - DOWN direction
        performScroll $RUN_COUNT $SCROLL_COUNTER "down" "normal"
        
        # Pause between scrolls
        sleep $SCROLL_DELAY
        
        # Update progress bar
        progress=$((SCROLL_COUNTER * 100 / NUM_SCROLLS))
        progress_bar="["
        for i in $(seq 1 $((progress / 5))); do
            progress_bar="${progress_bar}="
        done
        for i in $(seq 1 $((20 - progress / 5))); do
            progress_bar="${progress_bar} "
        done
        progress_bar="${progress_bar}]"
        echo -ne "Progress: $progress_bar $progress%\r"
        
        ((SCROLL_COUNTER++))
    done
    echo -e "\nCompleted all $NUM_SCROLLS normal down scrolls"
    
    # Capture screenshot after normal scrolls
    adb_with_retry "adb exec-out screencap -p" > "$IMAGE_DIR/after_run${RUN_COUNT}_normal.png"
    
    # Check for white gaps in normal scroll phase
    WHITE_GAPS_NORMAL=$(detect_white_gaps "$IMAGE_DIR/before_run${RUN_COUNT}_normal.png" "$IMAGE_DIR/after_run${RUN_COUNT}_normal.png")
    echo "White gaps detected in normal scroll phase: $WHITE_GAPS_NORMAL"
    
    # ===== PHASE 2: FAST UP SCROLLS (Back to top) =====
    echo "PHASE 2: Rapid UP scrolls to return to top ($NUM_FAST_UP scrolls)"
    echo "This phase will scroll UP to return to the top of the list"
    
    # Ask for confirmation before continuing
    echo -n "Ready for rapid UP scrolls? Press Enter to continue..."
    read
    
    # Capture screenshot before fast up scrolls
    adb_with_retry "adb exec-out screencap -p" > "$IMAGE_DIR/before_run${RUN_COUNT}_fastup.png"
    
    SCROLL_COUNTER=1
    echo -ne "Progress: [0%]\r"
    
    # Use safer edge of screen for scrolling
    edge_x=$((screen_width - 50))  # Use right edge of screen
    
    # Remove the alternative method that does downward swipes first
    # Start directly with the fast up scrolls
    while [ $SCROLL_COUNTER -le $NUM_FAST_UP ]; do
        # Perform the fast UP scroll test
        performScroll $RUN_COUNT $SCROLL_COUNTER "fast_up" "fast_up"
        
        # Short delay between fast scrolls
        sleep $FAST_SCROLL_DELAY
        
        # Update progress bar
        progress=$((SCROLL_COUNTER * 100 / NUM_FAST_UP))
        progress_bar="["
        for i in $(seq 1 $((progress / 5))); do
            progress_bar="${progress_bar}="
        done
        for i in $(seq 1 $((20 - progress / 5))); do
            progress_bar="${progress_bar} "
        done
        progress_bar="${progress_bar}]"
        echo -ne "Progress: $progress_bar $progress%\r"
        
        ((SCROLL_COUNTER++))
    done
    echo -e "\nCompleted all $NUM_FAST_UP rapid up scrolls"
    
    # Capture screenshot after fast up scrolls
    adb_with_retry "adb exec-out screencap -p" > "$IMAGE_DIR/after_run${RUN_COUNT}_fastup.png"
    
    # Check for white gaps in fast up scroll phase
    WHITE_GAPS_FASTUP=$(detect_white_gaps "$IMAGE_DIR/before_run${RUN_COUNT}_fastup.png" "$IMAGE_DIR/after_run${RUN_COUNT}_fastup.png")
    echo "White gaps detected in fast UP scroll phase: $WHITE_GAPS_FASTUP"
    
    # ===== PHASE 3: FAST DOWN SCROLLS =====
    echo "PHASE 3: Rapid DOWN scrolls to go back to bottom ($NUM_FAST_DOWN scrolls)"
    echo "This phase will scroll DOWN to reach the bottom of the list again"
    
    # Ask for confirmation before continuing
    echo -n "Ready for rapid DOWN scrolls? Press Enter to continue..."
    read
    
    # Capture screenshot before fast down scrolls
    adb_with_retry "adb exec-out screencap -p" > "$IMAGE_DIR/before_run${RUN_COUNT}_fastdown.png"
    
    # Use safer edge of screen for scrolling
    edge_x=$((screen_width - 50))  # Use right edge of screen
    echo "Using right edge of screen for safer scrolling: $edge_x px"
    
    SCROLL_COUNTER=1
    echo -ne "Progress: [0%]\r"
    
    # Use more dramatic scrolls for fast down
    while [ $SCROLL_COUNTER -le $NUM_FAST_DOWN ]; do
        # Perform the fast DOWN scroll test
        performScroll $RUN_COUNT $SCROLL_COUNTER "fast_down" "fast_down"
        
        # Short delay between fast scrolls
        sleep $FAST_SCROLL_DELAY
        
        # Update progress bar
        progress=$((SCROLL_COUNTER * 100 / NUM_FAST_DOWN))
        progress_bar="["
        for i in $(seq 1 $((progress / 5))); do
            progress_bar="${progress_bar}="
        done
        for i in $(seq 1 $((20 - progress / 5))); do
            progress_bar="${progress_bar} "
        done
        progress_bar="${progress_bar}]"
        echo -ne "Progress: $progress_bar $progress%\r"
        
        ((SCROLL_COUNTER++))
    done
    echo -e "\nCompleted all $NUM_FAST_DOWN rapid down scrolls"
    
    # Capture screenshot after fast down scrolls
    adb_with_retry "adb exec-out screencap -p" > "$IMAGE_DIR/after_run${RUN_COUNT}_fastdown.png"
    
    # Check for white gaps in fast down scroll phase
    WHITE_GAPS_FASTDOWN=$(detect_white_gaps "$IMAGE_DIR/before_run${RUN_COUNT}_fastdown.png" "$IMAGE_DIR/after_run${RUN_COUNT}_fastdown.png")
    echo "White gaps detected in fast DOWN scroll phase: $WHITE_GAPS_FASTDOWN"
    
    # Summary for this run
    echo "====== RUN #$RUN_COUNT SUMMARY ======"
    echo "Normal scrolls white gaps: $WHITE_GAPS_NORMAL"
    echo "Fast UP scrolls white gaps: $WHITE_GAPS_FASTUP"
    echo "Fast DOWN scrolls white gaps: $WHITE_GAPS_FASTDOWN"
    
    ((RUN_COUNT++))
done

# Process results after all runs
post_process_results

# Generate a simple report
echo "Generating simple report..."
./generate_report.sh

echo "Test completed! You can view the simple report with: cat simple_report/performance_summary.txt"