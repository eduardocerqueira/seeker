#date: 2024-12-19T17:03:40Z
#url: https://api.github.com/gists/f4aea3e8c09363bf4924b8bd2f80895d
#owner: https://api.github.com/users/jaggzh

#!/usr/bin/env python3
# Copyleft 2024 jaggz.h {over yonder at} gmail.com
# jaggzh @ reddit

# Usage:
# python plotter.py -i defocus.ods -o defocus.png

# Gist url for this script:
# https://gist.github.com/jaggzh/f4aea3e8c09363bf4924b8bd2f80895d

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from odf.opendocument import load
from odf.table import Table, TableRow, TableCell
from odf.text import P
import io
from bansi import * # Keep for perdy colors (eg. red, bred, cya, bcya, bgred, bgblu, yel, whi, rst)
import math
import numpy as np

debug_rows = 10 # Number of rows for debug display
np.random.seed(43)

def read_ods_sheet(file_path):
    print(f"Loading ODS file: {file_path}")
    doc = load(file_path)
    sheets = {}

    for sheet in doc.spreadsheet.getElementsByType(Table):
        sheet_name = sheet.getAttribute("name")
        print(f"Processing sheet: {sheet_name}")
        if sheet_name.startswith("#"):
            print(f"Ignoring sheet: {sheet_name}")
            continue
        rows = []
        for row_idx, row in enumerate(sheet.getElementsByType(TableRow)):
            cells = []
            for cell_idx, cell in enumerate(row.getElementsByType(TableCell)):
                # Check for numerical cells by `value-type`
                value_type = cell.attributes.get(('urn:oasis:names:tc:opendocument:xmlns:office:1.0', 'value-type'), None)
                if value_type == "float":
                    value = cell.attributes.get(('urn:oasis:names:tc:opendocument:xmlns:office:1.0', 'value'), None)
                    try:
                        value = float(value)  # Ensure it's a valid float
                    except ValueError:
                        print(f"Invalid float value at [{row_idx}][{cell_idx}]: {value}")
                        value = None
                else:
                    # Fall back to textual content if not numerical
                    value = cell.firstChild.data.strip() if cell.firstChild and cell.firstChild.nodeType == cell.TEXT_NODE else ""

                # Debug information with colors
                buffer = io.StringIO()
                cell.toXml(0, buffer)
                raw_content = buffer.getvalue()
                buffer.close()
                print(f"  Row/Cell [{bmag}{row_idx}{rst}][{yel}{cell_idx}{rst}] {bbla}Raw cell content:{rst} {raw_content} | Value type: {bgblu}{whi}{value_type}{rst} | Extracted value: {bgblu}{whi}{value}{rst}")

                cells.append(value)
            rows.append(cells)
        print(f"Raw data from sheet {sheet_name} (first {debug_rows} rows):\n{rows[:debug_rows]}")
        sheets[sheet_name] = pd.DataFrame(rows)

    return sheets

def clean_dataframes(sheets):
    processed_sheets = {}
    for name, df in sheets.items():
        print(f"Cleaning data from sheet: {name}")
        if df.empty:
            print(f"Sheet {name} is empty. Skipping.")
            continue
        try:
            # Detect the first numeric row
            numeric_start = df.apply(lambda row: row.apply(lambda x: isinstance(x, (int, float))).all(), axis=1).idxmax()
            df = df.iloc[numeric_start:]  # Skip non-numeric rows
            print(f"Data after skipping to numeric rows (first {debug_rows} rows):\n{df.head(debug_rows)}")

            # Reset column names explicitly
            df.columns = ["Diopter", "LogMAR"] + list(df.columns[2:])  # Only set the first two columns; keep the rest if any
            print(f"Set column names: {df.columns.tolist()}")

            # Convert to numeric and drop NaNs
            df = df[["Diopter", "LogMAR"]].apply(pd.to_numeric, errors='coerce').dropna()
            print(f"Data after numeric conversion from {name}:\n{df.head(debug_rows)}")

            if not df.empty:
                processed_sheets[name] = df
                print(f"Processed sheet {name} with {len(df)} rows.")
            else:
                print(f"Sheet {name} has no valid rows. Skipping.")
        except Exception as e:
            print(f"Error processing sheet {name}: {e}")
    return processed_sheets

def plot_sheets(processed_sheets, output_path, width, height):
    print("Starting to plot sheets...")
    plt.figure(figsize=(width / 100, height / 100))

    # Define line styles and thicknesses
    def rf(n,m):
        return np.random.uniform(n,m)

    mn=1.8
    mx=4.5
    plot_styles = [
        {"style": "solid", "lw": rf(mn,mx)},
        {"style": "dotted", "lw": rf(mn,mx)},
        {"style": "dashed", "lw": rf(mn,mx)},
        {"style": "dashdot", "lw": rf(mn,mx)},
        {"style": (0,(1,1)), "lw": rf(mn,mx)},
        {"style": (5,(10,3)), "lw": rf(mn,mx)},
        {"style": (0,(5,5)), "lw": rf(mn,mx)},
        {"style": (0,(5,8,5,8)), "lw": rf(mn,mx)},
    ]

    # Determine the overall X-axis range from all sheets
    x_min_data = min(df["Diopter"].min() for df in processed_sheets.values())
    x_max_data = max(df["Diopter"].max() for df in processed_sheets.values())

    # Round x_min and x_max to the nearest 0.25
    major_step = 1.0
    minor_step = 0.5
    extra_minor_step = 0.25

    x_min = round_down(x_min_data, extra_minor_step)
    x_max = round_up(x_max_data, extra_minor_step)

    print(f"X-axis range: {x_min_data} to {x_max_data}, rounded to {x_min} to {x_max}")

    # Cycle through plot styles
    style_count = len(plot_styles)
    for idx, (name, df) in enumerate(processed_sheets.items()):
        print(f"Plotting sheet: {name}")
        print(f"Data being plotted from {name}:\n{df}")
        
        # Get the current style by cycling through the list
        style = plot_styles[idx % style_count]

        # Plot the data using the selected style
        plt.plot(
            df["Diopter"],
            df["LogMAR"],
            linestyle=style["style"],
            linewidth=style["lw"],
            label=name,
        )

    if not processed_sheets:
        print("No data to plot. Exiting.")
        return

    # Set axis labels
    plt.xlabel("Diopter")
    plt.ylabel("LogMAR")

    # Add vertical lines for X-axis (major: every 1.0, minor: every 0.5, extra-minor: every 0.25)
    for x in range(int(x_min), int(x_max) + 1, int(major_step)):
        plt.axvline(x, color="#555", linestyle="-", linewidth=0.7)

    for x in [v for v in frange(x_min, x_max, minor_step) if v % major_step != 0]:
        plt.axvline(x, color="#999", linestyle="--", linewidth=0.5)

    for x in [v for v in frange(x_min, x_max, extra_minor_step) if v % minor_step != 0]:
        plt.axvline(x, color="#bbf", linestyle=":", linewidth=0.3)

    # Horizontal grid lines for Y-axis
    plt.gca().yaxis.set_major_locator(plt.MultipleLocator(0.1))
    plt.gca().yaxis.set_minor_locator(plt.MultipleLocator(0.025))
    plt.grid(visible=True, which="major", axis="y", color="#555", linestyle="-", linewidth=0.7)
    plt.grid(visible=True, which="minor", axis="y", color="#bbf", linestyle="--", linewidth=0.5)

    # Invert axes explicitly
    plt.gca().invert_xaxis()  # Flip the X-axis
    plt.gca().invert_yaxis()  # Flip the Y-axis

    plt.title("Combined Plots")
    plt.legend()
    plt.tight_layout()
    print(f"Saving plot to: {output_path}")
    plt.savefig(output_path)
    plt.close()

def round_down(value, step):
    return math.floor(value / step) * step

def round_up(value, step):
    return math.ceil(value / step) * step

# Helper function to generate a range of float values
def frange(start, stop, step):
    while start < stop:
        yield round(start, 10)  # Avoid floating-point precision issues
        start += step

def main():
    parser = argparse.ArgumentParser(description="Generate plots from .ods file sheets.")
    parser.add_argument("-i", "--input", required=True, help="Input .ods file")
    parser.add_argument("-o", "--output", required=True, help="Output .png file")
    parser.add_argument("-W", "--width", type=int, default=1800, help="Width of the output image (default: 1000)")
    parser.add_argument("-H", "--height", type=int, default=800, help="Height of the output image (default: 600)")

    args = parser.parse_args()

    # Read and process the .ods file
    sheets = read_ods_sheet(args.input)
    processed_sheets = clean_dataframes(sheets)

    # Plot the processed sheets
    plot_sheets(processed_sheets, args.output, args.width, args.height)

if __name__ == "__main__":
    main()
