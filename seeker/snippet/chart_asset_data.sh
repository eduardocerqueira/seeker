#date: 2024-11-06T16:51:52Z
#url: https://api.github.com/gists/6afa4422df45c3edcdd50868cc9c363d
#owner: https://api.github.com/users/silver-dragon

#!/bin/bash

# Define input CSV file and temporary file for formatted data
input_file="assets_data.csv"
temp_file="total_asset_data_for_plot.txt"
temp_file2="cash_data_for_plot.txt"
output_png="total_assets.png"
output_png2="diamonds_on_hand.png"


# Extract date (without time) and total_assets columns for gnuplot
awk -F, 'NR > 1 { print substr($1, 1, 10), $4 }' "$input_file" > "$temp_file"

# Extract date (without time) and diamonds columns for gnuplot
awk -F, 'NR > 1 { print substr($1, 1, 10), $3 }' "$input_file" > "$temp_file2"

# Debugging Step: Print the first few lines of the temporary file to verify the output
# echo "Sample data from $temp_file2:"
# head -n 5 "$temp_file2"

# Plot with gnuplot in ASCII mode
gnuplot <<- EOF
    set datafile separator " "
    set xdata time
    set timefmt "%Y-%m-%d"
    set format x "%m-%d"               # Display only month and day on x-axis
    set format y "%.0f"                # Display y-axis as plain numbers, no scientific notation
    set title "Total Assets Over Time"
    set xlabel "Date"
    set ylabel "Total Assets"
    set grid
    set terminal dumb size 80,20       # ASCII output, size 80x20
    set autoscale xfixmin
    set autoscale xfixmax
    plot "$temp_file" using 1:2 with lines notitle
EOF

# Plot with gnuplot in ASCII mode
gnuplot <<- EOF
    set datafile separator " "
    set xdata time
    set timefmt "%Y-%m-%d"
    set format x "%m-%d"               # Display only month and day on x-axis
    set format y "%.0f"                # Display y-axis as plain numbers, no scientific notation
    set title "On Hand Diamonds Over Time"
    set xlabel "Date"
    set ylabel "Diamonds"
    set grid
    set terminal dumb size 80,20       # ASCII output, size 80x20
    set autoscale xfixmin
    set autoscale xfixmax
    plot "$temp_file2" using 1:2 with lines notitle
EOF

# Plot with gnuplot in PNG format (transparent background, white text, orange plot line)
gnuplot <<- EOF
    set datafile separator " "
    set xdata time
    set timefmt "%Y-%m-%d"
    set format x "%m-%d"               # Display only month and day on x-axis
    set format y "%.0f"                # Display y-axis as plain numbers, no scientific notation
    set title "Total Assets Over Time" font ",14" textcolor "#FFFFFF"
    set xlabel "Date" font ",12" textcolor "#FFFFFF"
    set ylabel "Total Assets" font ",12" textcolor "#FFFFFF"
    set grid linecolor rgb "#FFFFFF"   # White grid lines
    set xtics textcolor rgb "#FFFFFF"      # White x-axis labels
    set ytics textcolor rgb "#FFFFFF"      # White y-axis labels
    set border linecolor rgb "#FFFFFF" # White border
    set terminal pngcairo size 400,300 enhanced font 'Arial,10' background rgb "black" # BLACK background
    set output "$output_png"
    set style line 1 linecolor rgb "#FE6A00" linetype 1 linewidth 2  # Orange plot curve (#FE6A00)
    set autoscale xfixmin
    set autoscale xfixmax
    plot "$temp_file" using 1:2 with lines notitle ls 1
EOF


gnuplot <<- EOF
    set datafile separator " "
    set xdata time
    set timefmt "%Y-%m-%d"
    set format x "%m-%d"               # Display only month and day on x-axis
    set format y "%.0f"                # Display y-axis as plain numbers, no scientific notation
    set title "Diamonds On Hand Over Time" font ",14" textcolor "#FFFFFF"
    set xlabel "Date" font ",12" textcolor "#FFFFFF"
    set ylabel "Diamonds" font ",12" textcolor "#FFFFFF"
    set grid linecolor rgb "#FFFFFF"   # White grid lines
    set xtics textcolor rgb "#FFFFFF"      # White x-axis labels
    set ytics textcolor rgb "#FFFFFF"      # White y-axis labels
    set border linecolor rgb "#FFFFFF" # White border
    set terminal pngcairo size 400,300 enhanced font 'Arial,10' background rgb "black" # BLACK background
    set output "$output_png2"
    set style line 1 linecolor rgb "#2CB9CF" linetype 1 linewidth 2  # teal plot curve
    set autoscale xfixmin
    set autoscale xfixmax
    plot "$temp_file2" using 1:2 with lines notitle ls 1
EOF

# Copy the generated PNG file to the webserver via SSH
scp "$output_png" "$output_png2" user@yourserver.com:/path/to/your/http/folder/

# Clean up temporary file
rm "$temp_file"
rm "$temp_file2"
