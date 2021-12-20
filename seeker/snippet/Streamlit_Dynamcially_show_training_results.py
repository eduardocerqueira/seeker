#date: 2021-12-20T17:18:30Z
#url: https://api.github.com/gists/462348717020739ac70a24fcf165d002
#owner: https://api.github.com/users/dormeir999

# Helper functions

def process_running_without_jupyter():
    """
    list all child processes name that are running now and are not jupyter lab (the free bench page process)
    Returns: list of processes names
    """
    try:
        return [proc.name() for proc in psutil.Process(os.getpid()).children() if "jupyter lab --allow-root" not in proc.cmdline()]
    except psutil.NoSuchProcess:
        return []
    
def last_changed_file(dir_path=None, ends_with="", report_error=True):
    try:
        if dir_path is None:
            dir_path = os.getcwd()
        list_of_files = glob.glob(dir_path + '/*')
        if len(ends_with) > 0:
            list_of_files = [file for file in list_of_files if file.endswith(ends_with)]
        try:
            return max(list_of_files, key=os.path.getmtime)
        except ValueError:
            if report_error:
                st.info("No files in folder")
            list_of_files = glob.glob(dir_path + '/*')
            return None
    except NotADirectoryError:
        st.info("Choose a directory")
        
def is_dir_not_empty(folder):
    if os.path.exists(folder):
        if len(os.listdir(folder)):
            return True
        return False
    return False

def run_script(script_type, state, process_table_sidebar=None, timestamp=True):
    if script_type == 'validation_inside_modeling':
        os.chdir(configs['LSTM_DEFAULT_DIR'])
        state.chosen_weights_abs_folder = os.path.join(state.general_weights_dir, state.chosen_weights_folder)
        command = rf"activate.bat {state.LSTM_env} & python {state.LSTM_dir}\validation_graphs.py --logs_path={state.running_model_logs} --save_path={state.validation_graphs_dir_train}" + output_to_log
        try:
            subprocess.Popen(command)
        except:
            st.error(f"Couldn't run {script_type}:")
            e = sys.exc_info()
            st.error(e)
        updated_processes_table(process_table_sidebar, state)

def validation_inside_modeling(state, process_table_sidebar, key=""):
    st.header(":chart_with_downwards_trend: Graphs")
    state.last_changed_dir_abs_path = last_changed_file(state.general_weights_dir)
    last_changed_dir = os.path.basename(state.last_changed_dir_abs_path)
    state.validation_graphs_dir_train = os.path.join(configs['VALIDATION_GRPAHS_DIR'],
                                                     state.chosen_weights_folder)
    wait_message = st.empty()
    model_validation_logs_path = os.path.join(os.path.dirname(state.weights_init_with), "validation")
    in_progress_message = st.empty()
    refresh_message = st.empty()
    if "Train" in state.process_df.name.tolist():
        state.refresh_interval = st.number_input("Refresh interval",
                                                 value=state.refresh_interval if state.refresh_interval else 60)
    else:
        st.info("No training in progress")
    model_message = st.empty()
    images_box = st.empty()
    logs_messages = st.empty()
    while "Train" in state.process_df.name.tolist():
        state.running_model_logs = last_changed_file(state.general_weights_dir)
        state.running_model_name = os.path.basename(state.running_model_logs)
        state.validation_graphs_dir_train = os.path.join(configs['VALIDATION_GRPAHS_DIR'],
                                                         state.running_model_name)
        model_message.subheader(f"Graphs of {state.running_model_name}:")
        in_progress_message.success('Training in progress')

        if is_dir_not_empty(model_validation_logs_path):
            run_script("validation_inside_modeling", state, process_table_sidebar, timestamp=False)
        else:
            no_val_logs_message = f"Creating validation logs for {state.chosen_weights_folder}"
            logs_messages.info(no_val_logs_message)

        if is_dir_not_empty(state.validation_graphs_dir_train):
            os.chdir(state.validation_graphs_dir_train)
            graphs = os.listdir(os.getcwd())
            modification_time = [cpu_time_to_calender(os.path.getmtime(os.path.join(os.getcwd(), graph))) for graph
                                 in graphs]
            images_box.image(graphs, caption=modification_time)
        else:
            logs_messages.info(
                f"{os.path.basename(state.validation_graphs_dir_train)} has no validation logs (didn't finish one "
                f"epoch of training)")
            last_model_val_logs_path = os.path.join(state.last_changed_dir_abs_path, "validation")
            if os.path.exists(last_model_val_logs_path) and st.button(
                    f'See "{os.path.basename(state.last_changed_dir_abs_path)}" graphs?'):
                run_script("validation_inside_modeling", state, process_table_sidebar, timestamp=False)
                os.chdir(state.validation_graphs_dir_train)
                graphs = os.listdir(os.getcwd())
                modification_time = [cpu_time_to_calender(os.path.getmtime(os.path.join(os.getcwd(), graph)))
                                     for graph in graphs]
                try:
                    images_box.image(graphs, modification_time)
                except FileNotFoundError:
                    images_box.image(graphs, modification_time)

        for i in range(state.refresh_interval):
            time.sleep(1)
            refresh_message.info(f"Refreshing in {state.refresh_interval - i} seconds")
            
def logging_while_running(state, key="", log_lines_to_show=1):
    st.header(":scroll: Logs")
    no_training_message = st.empty()
    list_of_files = glob.glob(
        configs[
            'APP_BASE_DIR'] + '/logs/models_logs/modeling*')  # * - all if need specific format then *.csv
    state.app_log_file = max(list_of_files, key=os.path.getmtime)
    epoch_box = st.empty()
    logs_box = st.empty()
    no_training_message.info('No training in progress')
    if st.button(f"Download {os.path.basename(state.app_log_file)}", key="mainbar_log"):
        with open(state.app_log_file) as f:
            content = f.readlines()
        df = pd.DataFrame(content)  # , columns=["Col1", "Col2", "Col3"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{state.app_log_file}">{state.app_log_file}</a>'
        st.markdown(href, unsafe_allow_html=True)
    while (process_running_without_jupyter()):
        no_training_message.empty()
        time.sleep(1)
        with open(state.app_log_file) as f:
            lines = f.read().splitlines()
            epoch = [line for line in lines if "Epoch" in line]
            if epoch:
                epoch = epoch[-1]
                epoch_box.success(epoch)
            last_lines = lines[-log_lines_to_show:]
            logs_box.code(last_lines[0])
            
# Implementation

inspect_model_method = st.radio(label='', options=['Logs', 'Graphs'], index=0)

with col2:
    if inspect_model_method == 'Logs':
        logging_while_running(state, key="logging_while_running")
    elif inspect_model_method == 'Graphs':
        validation_inside_modeling(state, process_table_sidebar, key="graphs")