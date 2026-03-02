#date: 2026-03-02T17:38:53Z
#url: https://api.github.com/gists/eee9a448be7287a024d260ec80751120
#owner: https://api.github.com/users/fomightez

# meant to be run with `uv run https://gist.githubusercontent.com/fomightez/eee9a448be7287a024d260ec80751120/raw/30949804da8ea3f2ab4f08bf18a6bfd390b02574/streamlined_timestamp_start_end_miner_checker.py out.txt`, or similar
# This handles evaluating date timestamp info in start and end timestamps of a pipeline.
#####*****------------------------------------------------------------*****#####
# This is meant to use with `uv` to run. 
# First install `uv` with `pip install uv` then run `!uv run {script_url} {input_text_filepath}` where defined those variables prior
#-------------------------------------------------------------#
# Times printed for now. (Make a dataframe?)
#-------------------------------------------------------------#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
# ]
# ///
def collect_time_info(input_text_filepath):
    '''
    return the time difference from gleaned from start and end timestamps in the corresponding 
    `.out` that I typically ensure get generated when running tasks on OSG
    '''
    with open(input_text_filepath, 'r') as thelog_stdout_file:
        std_out_string=thelog_stdout_file.read()
    # with std_out log read in, parse it for the informaiton in the three timestamps
    start_ts = std_out_string.split('Current timestamp at start: ')[1].split('\n')[0].strip()
    end_main_event_ts = std_out_string.split('Current timestamp after: ')[1].split('\n')[0].strip()
    # determine time duration between events in minutes
    # For Total Time
    minutes_diff = round((datetime.strptime(end_main_event_ts, "%Y-%m-%d_%H-%M-%S") - datetime.strptime(start_ts, "%Y-%m-%d_%H-%M-%S")).total_seconds() / 60)
    hours = int(minutes_diff // 60)
    mins = int(minutes_diff % 60)
    if minutes_diff > 60:
        print(f"Total time processing run: {minutes_diff}m ({hours}h {mins}m)")
    else:
        print(f"Total time processing run: {minutes_diff}m")


if __name__ == "__main__":
    import sys
    from datetime import datetime
    try:
        input_text_filepath = sys.argv[1]
    except IndexError:
        import rich
        rich.print("\n[bold red]I suspect you forgot to specify the file to read?[/bold red]\n **EXITING !!**[/bold red]\n"); sys.exit(1)
    collect_time_info(input_text_filepath)