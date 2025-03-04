#date: 2025-03-04T16:58:03Z
#url: https://api.github.com/gists/a5bee20de7749f3a9841ffdefeea6e0a
#owner: https://api.github.com/users/Magnus167

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dash_utils

import concurrent.futures as cf


def calc_errors_page():
    st.title("Calc-Errors report")
    st.write("Calc-Errors report for PROD, UAT, and DEV environments.")

    # Create tabs for each environment
    tab_prod, tab_uat, tab_dev = st.tabs(["PROD", "UAT", "DEV"])
    tabs = {"PROD": tab_prod, "UAT": tab_uat, "DEV": tab_dev}

    # Create placeholders for each tab so we can update them later
    _ = {env: tabs[env].empty() for env in ["PROD", "UAT", "DEV"]}

    # Launch database calls concurrently using ThreadPoolExecutor
    with cf.ThreadPoolExecutor(max_workers=3) as executor:
        # Map each environment to a future
        future_to_env = {
            executor.submit(dash_utils.get_calc_error_report, env=env): env
            for env in ["PROD", "UAT", "DEV"]
        }
        for future in cf.as_completed(future_to_env):
            env = future_to_env[future]
            try:
                data = future.result()
            except Exception as e:
                data = f"Error fetching data for {env}: {e}"
            # Update the corresponding tab with the results
            with tabs[env]:
                st.subheader(f"Calc-Errors Report ({env})")
                if isinstance(data, pd.DataFrame):
                    st.dataframe(data)
                else:
                    st.write(data)


def collect_errors_page():
    st.title("Collect-Errors report")
    st.write("Collect-Errors report for PROD, UAT, and DEV environments.")

    # Create tabs for each environment
    tab_prod, tab_uat, tab_dev = st.tabs(["PROD", "UAT", "DEV"])
    tabs = {"PROD": tab_prod, "UAT": tab_uat, "DEV": tab_dev}

    # Create placeholders for each tab so we can update them later
    _ = {env: tabs[env].empty() for env in ["PROD", "UAT", "DEV"]}

    # Launch database calls concurrently using ThreadPoolExecutor
    with cf.ThreadPoolExecutor(max_workers=3) as executor:
        # Map each environment to a future
        future_to_env = {
            executor.submit(dash_utils.get_collect_error_report, env=env): env
            for env in ["PROD", "UAT", "DEV"]
        }
        for future in cf.as_completed(future_to_env):
            env = future_to_env[future]
            try:
                data = future.result()
            except Exception as e:
                data = f"Error fetching data for {env}: {e}"
            # Update the corresponding tab with the results
            with tabs[env]:
                st.subheader(f"Collect-Errors Report ({env})")
                if isinstance(data, pd.DataFrame):
                    st.dataframe(data)
                else:
                    st.write(data)


def discontinued_series_page():
    st.title("Discontinued-Series report")
    st.write("Discontinued-Series report for PROD, UAT, and DEV environments.")

    # Create tabs for each environment
    tab_prod, tab_uat, tab_dev = st.tabs(["PROD", "UAT", "DEV"])
    tabs = {"PROD": tab_prod, "UAT": tab_uat, "DEV": tab_dev}

    # Create placeholders for each tab so we can update them later
    _ = {env: tabs[env].empty() for env in ["PROD", "UAT", "DEV"]}

    # Launch database calls concurrently using ThreadPoolExecutor
    with cf.ThreadPoolExecutor(max_workers=3) as executor:
        # Map each environment to a future
        future_to_env = {
            executor.submit(dash_utils.get_discontinued_series_report, env=env): env
            for env in ["PROD", "UAT", "DEV"]
        }
        for future in cf.as_completed(future_to_env):
            env = future_to_env[future]
            try:
                data = future.result()
            except Exception as e:
                data = f"Error fetching data for {env}: {e}"
            # Update the corresponding tab with the results
            with tabs[env]:
                st.subheader(f"Discontinued-Series Report ({env})")
                if isinstance(data, pd.DataFrame):
                    st.dataframe(data)
                else:
                    st.write(data)


def chart_page():
    st.title("Diff Chart Page")
    st.write(
        "This chart shows the difference between the indicator published today and yesterday."
    )
    tickers_list = dash_utils.get_tickers_list()
    ticker = st.selectbox("Select Ticker", tickers_list)
    curr_df, prev_df = dash_utils.get_ticker_files(ticker)
    old_name, new_name = f"{ticker} (old)", f"{ticker} (new)"
    curr_df = curr_df.set_index("real_date")["value"].rename(new_name)
    prev_df = prev_df.set_index("real_date")["value"].rename(old_name)
    diff_df = curr_df - prev_df
    data = pd.concat([prev_df, curr_df, diff_df], axis=1)
    data = data.fillna(method="ffill")
    st.line_chart(diff_df)


def sources_chart_page():

    st.markdown(
        """
    <iframe 
        src="https://macrosynergy.github.io/msy-content-links/"
        width="1500" 
        height="1500" 
        style="border:none;">
    </iframe>
    """,
        unsafe_allow_html=True,
    )


def buttons_page():
    st.title("Buttons Page")
    st.write("Here are some buttons with links.")
    st.button(
        "JPMaQS on JPMM",
        on_click=lambda: st.write("[JPMaQS on JPMM](https://www.jpmm.com/#jpmaqs)"),
    )
    st.button(
        "JPMaQS Confluence",
        on_click=lambda: st.write(
            "[JPMaQS Confluence](https://example.com/confluence)"
        ),
    )
    st.button(
        "Macrosynergy Package docs",
        on_click=lambda: st.write(
            "[Macrosynergy Package docs](https://docs.macrosynergy.com)"
        ),
    )


def multi_charts_page():
    st.title("Multi-Chart Page")
    st.write("Select a chart type from the dropdown.")
    chart_type = st.selectbox(
        "Select Chart Type", ["Line Chart", "Bar Chart", "Scatter Plot"]
    )
    data = pd.DataFrame(np.random.randn(20, 2), columns=["X", "Y"])
    if chart_type == "Line Chart":
        st.line_chart(data)
    elif chart_type == "Bar Chart":
        st.bar_chart(data)
    elif chart_type == "Scatter Plot":
        st.scatter_chart(data)


PAGE_KEYS = {
    "Buttons": buttons_page,
    "Multi-Charts": multi_charts_page,
    "Diff Chart": chart_page,
    "Sources Chart": sources_chart_page,
    "Calc Errors": calc_errors_page,
    "Collect Errors": collect_errors_page,
    "Discontinued Series": discontinued_series_page,
}


def main():
    st.set_page_config(page_title="JPMaQS Dashboard", layout="wide")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        list(PAGE_KEYS.keys()),
    )
    assert page in PAGE_KEYS, f"Invalid page: {page}"
    PAGE_KEYS[page]()


if __name__ == "__main__":
    main()