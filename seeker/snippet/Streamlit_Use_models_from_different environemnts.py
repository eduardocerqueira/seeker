#date: 2021-12-20T17:17:33Z
#url: https://api.github.com/gists/7bc8deec551452a74e95d2e3e93ce1b3
#owner: https://api.github.com/users/dormeir999

# Helper functions

def now_string(seconds=False, microsec=False, only_date=False, log_timestamp=False):
    now = datetime.datetime.now()
    if log_timestamp:
        return str(now.day) + "-" + str(now.month) + "-" + str(now.year) + ' ' + str(now.hour) + ":" + str(
            now.minute) + ":" + str(now.second) + "." + str(now.microsecond)[:2] + " ~~ "
    if only_date:
        return str(now.day) + "_" + str(now.month) + "_" + str(now.year)
    now_str = str(now.day) + "_" + str(now.month) + "_" + str(now.year) + '_' + str(now.hour) + "_" + str(now.minute)
    if seconds:
        now_str += "_" + str(now.second)
    if microsec:
        now_str += "_" + str(now.microsecond)
    return now_str
  
def write_command_str_with_timestamp_to_log(log, command):
    try:
        with open(log, "a") as myfile:
            timestamp = now_string(log_timestamp=True)
            myfile.write(timestamp + command + "\n")
    except PermissionError:
        st.info("Couldn't timestamp log, Permission error")
        
# Implementation

if script_type == 'modeling_LSTM':
    script_name = 'Train.py'
    script_path = os.path.join(state.LSTM_dir, script_name)
    env_activate = rf"activate.bat {state.LSTM_env}"
    script_command = rf""" & python {script_path} """
    os.chdir(configs['LSTM_DEFAULT_DIR'])
    command = env_activate + script_command + state.maskrcnn_train_arguments + output_to_log

    write_command_str_with_timestamp_to_log(log, command)
    try:
        subprocess.Popen(command)
        updated_processes_table(process_table_sidebar, state)
        st.success('Started Training successfully')
    except:
        st.error(f"Couldn't train model {state.model}:")
        e = sys.exc_info()
        st.error(e)